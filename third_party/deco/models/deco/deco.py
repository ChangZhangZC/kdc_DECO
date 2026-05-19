import math
import torch
import einops
from torch import Tensor, nn
from torch.nn import functional as F
from models.deco.img_encoder import build_resnet_backbone
from models.deco.denoise_schedular import get_schedule
from models.deco.rope import apply_rotary_emb, RotaryPosEmbed


class RGBDepthCrossAttentionFusion(nn.Module):
    """ACT 风格 RGB-depth 双向 cross attention。

    输入和输出均为 token 序列 [B, L, dim]。阶段三第一版不把 RGB-D 压成
    单路 visual token，而是保留 fused_rgb / fused_depth 两路语义，继续适配
    DECO 原生两路视觉 token 假设。
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.rgb_to_depth_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.depth_to_rgb_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.rgb_norm = nn.LayerNorm(dim)
        self.depth_norm = nn.LayerNorm(dim)

    def forward(self, rgb_tokens, depth_tokens):
        if rgb_tokens.shape != depth_tokens.shape:
            raise ValueError(
                "RGB/depth token shapes must match before fusion, "
                f"got rgb={tuple(rgb_tokens.shape)}, depth={tuple(depth_tokens.shape)}"
            )

        fused_rgb, _ = self.rgb_to_depth_attn(
            query=rgb_tokens,
            key=depth_tokens,
            value=depth_tokens,
        )
        fused_depth, _ = self.depth_to_rgb_attn(
            query=depth_tokens,
            key=rgb_tokens,
            value=rgb_tokens,
        )
        fused_rgb = self.rgb_norm(rgb_tokens + fused_rgb)
        fused_depth = self.depth_norm(depth_tokens + fused_depth)
        return fused_rgb, fused_depth


class DECO(nn.Module):
    def __init__(
        self,
        act_dim,
        chunk_size,
        obs_state=True,
        use_tactile=False,
        plugin=False,
        plugin_rank=32,
        use_task_condition=False,
        num_tasks=10,
        inf_step=10,
        num_attn_blocks=6,
        heads=8,
        dim=512,
        rope_axes_dim=[256, 256],
        vision_backbone="resnet34",
        depth_backbone="resnet34",
    ):
        super().__init__()
        head_dim = dim // heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_state = obs_state
        self.use_tactile = use_tactile
        self.use_task_condition = use_task_condition
        self.inference_step = inf_step
        self.rope = RotaryPosEmbed(head_dim, rope_axes_dim)  # initial mrope embedding
        self.img_encoder = build_resnet_backbone(vision_backbone, in_channels=3)  # RGB encoder
        self.img_head = nn.Conv2d(512, dim, kernel_size=3, padding=1)
        self.depth_encoder = build_resnet_backbone(depth_backbone, in_channels=1)
        self.depth_head = nn.Conv2d(512, dim, kernel_size=3, padding=1)
        self.rgb_depth_fusion = RGBDepthCrossAttentionFusion(dim=dim, heads=heads)
        self.sync_depth_conv1_from_rgb()
        
        self.pos_idx_embedd = nn.Embedding(2, dim) # stream 0: RGB, stream 1: depth
        if self.obs_state:
            self.obs_encoder = nn.Sequential(
                nn.Linear(act_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim)
            )
        if self.use_tactile:
            # Kuavo 触觉已经在数据转换阶段提取为左右手各 15 维法向力。
            # wrapper / dataset 应先完成 DECO-style tactile max 归一化，再传入这里。
            self.tactile_dim_per_hand = 15
            self.tactile_condition_dim = 15 + 15 + 34
            self.gated = nn.Linear(self.tactile_condition_dim, self.tactile_condition_dim, bias=False)

            # positional embedding for tactile regions
            self.pos_tac_embedd = nn.Sequential(
                            timeEmb(dim),
                            nn.Linear(dim, dim * 4),
                            nn.Mish(),
                            nn.Linear(dim * 4, dim)
                        )

            # Kuavo tactile: (b, 15 + 15) --> (b, 34)
            self.tactile_encoder = nn.Sequential(
                nn.Linear(self.tactile_dim_per_hand * 2, 512),
                nn.Mish(),
                nn.Linear(512, 34),
            )
        
        if self.use_task_condition:
            self.task_encoder = nn.Embedding(num_tasks, dim)

        self.time_embedd = nn.Sequential(
            timeEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.action_embedd = nn.Parameter(torch.zeros(1, chunk_size, dim))  # learnable action positional embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim)
        )

        self.mmattn = nn.ModuleList([MMAttention(heads, dim, use_tactile, plugin, plugin_rank) for _ in range(num_attn_blocks)])  # joint attention blocks
        self.linear = nn.Linear(dim, act_dim) # final action prediction head

        if not plugin:
            self.initialize_weights()

    def forward(self, rgb, depth, obs=None, act=None, task_idx=None, tac1=None, tac2=None, action_mask=None, training=True):
        """
        Args:
            rgb: [B, 3, H, W], Kuavo 头部 RGB 图像
            depth: [B, 1, H, W] 或 [B, 3, H, W], Kuavo depth 图像；3 通道兼容存储会取第一通道
            obs: [B, 28]
            act: [B, chunk, 28]
            task_idx: [B, ]
            tac1: [B, 15], Kuavo 左手 DECO-style normalized tactile
            tac2: [B, 15], Kuavo 右手 DECO-style normalized tactile
            training: bool
        """
        # RGB-D encoding：输出仍保持两路视觉 token，前半为 RGB stream，后半为 depth stream。
        feat, image_rotary_emb = self.img_encoding(rgb, depth) # feat:[B, 2*img_seq_len, dim]
        if self.use_tactile:
            tactile = self.encode_kuavo_tactile(tac1, tac2)
        else:
            tactile = None

        if self.obs_state:
            obs = self.obs_encoder(obs)  # (b, act_dim) --> (b, dim)
        if self.use_task_condition: # onehot condition for different sub-tasks
            task_emb = self.task_encoder(task_idx) # (b, ) --> (b, dim)

        if training:
            t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))  # t in [0, 1] (b, )
            act, noise = self.add_noise(act, t)
            t = self.time_embedd(t)  # time embedding, (b, ) --> (b, dim)
            if self.obs_state:
                t = t + obs
            if self.use_task_condition:
                t = t + task_emb
            feat, act = self.atten_forward(feat, act, image_rotary_emb=image_rotary_emb, t=t, tactile=tactile)
            return act, noise

        else:
            sample = torch.randn(rgb.shape[0], self.chunk_size, self.act_dim).to(rgb.device)  # initial noisy action
            t = get_schedule(self.inference_step, self.chunk_size)  # get denoising schedule
            for t_curr, t_prev in zip(t[:-1], t[1:]):  # denoising loop
                t_vec = torch.full((rgb.shape[0],), t_curr, dtype=rgb.dtype, device=rgb.device)
                t_vec = self.time_embedd(t_vec)  # time embedding
                if self.obs_state:
                    t_vec = t_vec + obs
                if self.use_task_condition:
                    t_vec = t_vec + task_emb
                _, denoise_act = self.atten_forward(feat, sample, image_rotary_emb=image_rotary_emb, t=t_vec, tactile=tactile)
                # denoise_act : noise - action
                # t_prev - t_curr < 0 ---> -|△t|
                # action =  noise_act + |△t| * (action - noise)
                sample = sample + (t_prev - t_curr) * denoise_act

            return sample

    def img_encoding(self, rgb, depth):
        return self.rgbd_img_encoding(rgb, depth)

    def rgbd_img_encoding(self, rgb, depth):
        if rgb.ndim != 4 or depth.ndim != 4:
            raise ValueError(
                "RGB-D visual inputs must be 4D tensors [B, C, H, W], "
                f"got rgb={tuple(rgb.shape)}, depth={tuple(depth.shape)}"
            )
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError(
                "RGB and depth must share batch/spatial shape, "
                f"got rgb={tuple(rgb.shape)}, depth={tuple(depth.shape)}"
            )
        if rgb.shape[1] != 3:
            raise ValueError(f"RGB stream expects 3 channels, got {rgb.shape[1]}")

        depth = self.prepare_depth_input(depth)
        rgb_feat = self.img_head(self.img_encoder(rgb))
        depth_feat = self.depth_head(self.depth_encoder(depth))
        if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
            raise ValueError(
                "RGB/depth feature maps must share spatial shape before fusion, "
                f"got rgb={tuple(rgb_feat.shape)}, depth={tuple(depth_feat.shape)}"
            )

        rgb_tokens = einops.rearrange(rgb_feat, 'b c h w -> b (h w) c')
        depth_tokens = einops.rearrange(depth_feat, 'b c h w -> b (h w) c')
        fused_rgb_tokens, fused_depth_tokens = self.rgb_depth_fusion(rgb_tokens, depth_tokens)
        return self.pack_visual_token_sequences(
            fused_rgb_tokens,
            fused_depth_tokens,
            rgb_feat.shape[-2],
            rgb_feat.shape[-1],
            rgb.device,
        )

    def prepare_depth_input(self, depth):
        if depth.shape[1] == 1:
            return depth
        if depth.shape[1] == 3:
            # 当前 Kuavo-DECO 第一版 depth 为 3-channel uint8 兼容存储；
            # 三个通道由同一深度图 repeat 得到，因此模型侧取第一通道恢复 1-channel depth 语义。
            return depth[:, :1]
        raise ValueError(f"Depth stream expects 1 or 3 channels, got {depth.shape[1]}")

    def pack_visual_token_sequences(self, feat1, feat2, feat_h, feat_w, device):
        image_rotary_emb = self.rope(feat_h, feat_w)
        # stream_id=0 表示 Kuavo RGB；stream_id=1 表示 Kuavo depth。
        stream_id = torch.tensor([0] * feat1.shape[1] + [1] * feat2.shape[1]).to(device)
        stream_emb = self.pos_idx_embedd(stream_id).repeat(feat1.shape[0], 1, 1)
        feat = torch.cat([feat1, feat2], dim=1)  # (b, 2*seq_len, dim)
        feat = feat + stream_emb

        return feat, image_rotary_emb

    def encode_kuavo_tactile(self, tac1, tac2):
        if tac1 is None or tac2 is None:
            raise ValueError("use_tactile=True requires tac1 and tac2 tensors")
        if tac1.ndim != 2 or tac2.ndim != 2:
            raise ValueError(
                "Kuavo tactile inputs must be [B, 15], "
                f"got tac1={tuple(tac1.shape)}, tac2={tuple(tac2.shape)}"
            )
        if tac1.shape != tac2.shape or tac1.shape[-1] != self.tactile_dim_per_hand:
            raise ValueError(
                "Kuavo tactile expects left/right 15D tensors with the same shape, "
                f"got tac1={tuple(tac1.shape)}, tac2={tuple(tac2.shape)}"
            )

        tactile_emb = self.tactile_encoder(torch.cat([tac1, tac2], dim=-1))
        tactile = torch.cat([tac1, tac2, tactile_emb], dim=-1)
        tactile = tactile * torch.sigmoid(self.gated(tactile))
        tactile = self.pos_tac_embedd(tactile)  # (b, 64) --> (b, 64, dim)
        return tactile


    def atten_forward(self, img, act, image_rotary_emb, t, tactile=None):
        act = self.action_encoder(act)   # (b, seq_len, act_dim) --> (b, seq_len, dim)
        act = act + self.action_embedd   # add learnable action positional embedding

        for mma in self.mmattn:
            img, act = mma(img, act, t, image_rotary_emb, tactile) 

        act = self.linear(act)
        return img, act
    
    def add_noise(self, act: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(act).to(act.device)
        t = t.view(act.shape[0], 1, 1)
        act = (1 - t) * act + t * noise
        return act, noise

    def sync_depth_conv1_from_rgb(self):
        rgb_conv = self.img_encoder.conv1
        depth_conv = self.depth_encoder.conv1
        if rgb_conv.weight.shape[1] != 3 or depth_conv.weight.shape[1] != 1:
            raise ValueError(
                "RGB/depth conv1 channel mismatch, "
                f"rgb={tuple(rgb_conv.weight.shape)}, depth={tuple(depth_conv.weight.shape)}"
            )
        with torch.no_grad():
            depth_conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

# low-rank adapter module
class PI_Adapter(nn.Module): 
    def __init__(self, dim, out_dim, rank=32):
        super().__init__()

        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, out_dim)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        x = self.down(x)
        x = self.up(x)
        return x


class MMAttention(nn.Module):
    def __init__(self, heads=8, dim=512, use_tactile=False, plugin=False, plugin_rank=32):
        super().__init__()
        head_dim = dim // heads
        self.head_dim = dim // heads
        self.head = heads
        self.use_tactile = use_tactile
        self.plugin = plugin

        ### img projection
        self.img_bais = adaLN(dim)  # adaLN for time conditioning

        # attention
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.img_qknorm = QKNorm(head_dim)
        self.img_proj = nn.Linear(dim, dim)
        # mlp
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim*4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim*4, dim, bias=True),
        )
        
        ### action projection
        self.act_bais = adaLN(dim)  # adaLN for time conditioning

        # attention
        self.act_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_qkv = nn.Linear(dim, dim * 3)
        self.act_qknorm = QKNorm(head_dim)
        self.act_proj = nn.Linear(dim, dim)
        # mlp
        self.act_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_mlp = nn.Sequential(
            nn.Linear(dim, dim*4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim*4, dim, bias=True),
        )

        ### cross attention for tactile
        if self.use_tactile:
            self.tactile_key = nn.Linear(dim, dim)
            self.tactile_value = nn.Linear(dim, dim)
        
            if plugin:  # adapter modules for finetuning vision-based model
                self.img_qkv_pi = PI_Adapter(dim, dim*3, plugin_rank)
                self.img_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.img_mlp_pi = PI_Adapter(dim, dim, plugin_rank)

                self.act_qkv_pi = PI_Adapter(dim, dim*3,plugin_rank)
                self.act_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.act_mlp_pi = PI_Adapter(dim, dim, plugin_rank)


    def forward(self, img, act, t, image_rotary_emb, tactile=None):
        """
        Args:
            img: [B, 2*img_len, dim], 前半为 RGB stream，后半为 depth stream
            act: [B, chunk, dim]
            t: [B, dim]
            image_rotary_emb: tuple[Tensor, Tensor]
            tactile: [B, 64, dim]
        """
        total_img_len = img.shape[1]
        scale1_feat, shift1_feat, gate1_feat, scale2_feat, shift2_feat, gate2_feat = self.img_bais(t)  # adaLN
        img_norm = self.img_norm1(img)  # layer norm
        img_norm = (1 + scale1_feat) * img_norm + shift1_feat
        img_qkv = self.img_qkv(img_norm)
        if self.use_tactile and self.plugin:
            img_qkv += self.img_qkv_pi(img_norm)

        img_q, img_k, img_v = einops.rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.head, D=self.head_dim)
        img_q, img_k = self.img_qknorm(img_q, img_k, img_v)  # qk norm
        # q,k,v: [B, H, l1, D]
        feat_len = int(total_img_len / 2)
        # 前后两半共享同一组二维 RoPE；Kuavo RGB-D 模式下两半分别是 RGB stream 与 depth stream。
        img_q[:, :, :feat_len, :], img_k[:, :, :feat_len, :] = apply_rotary_emb(img_q[:, :, :feat_len, :], image_rotary_emb), apply_rotary_emb(img_k[:, :, :feat_len, :], image_rotary_emb)
        img_q[:, :, feat_len:, :], img_k[:, :, feat_len:, :] = apply_rotary_emb(img_q[:, :, feat_len:, :], image_rotary_emb), apply_rotary_emb(img_k[:, :, feat_len:, :], image_rotary_emb)
        

        scale1_act, shift1_act, gate1_act, scale2_act, shift2_act, gate2_act = self.act_bais(t) # adaLN
        act_norm = self.act_norm1(act)  # action is already added learnable pos emb
        act_norm = (1 + scale1_act) * act_norm + shift1_act
        act_qkv = self.act_qkv(act_norm)
        if self.use_tactile and self.plugin:
            act_qkv += self.act_qkv_pi(act_norm)

        act_q, act_k, act_v = einops.rearrange(act_qkv, "B L (K H D) -> K B H L D", K=3, H=self.head, D=self.head_dim)
        act_q, act_k = self.act_qknorm(act_q, act_k, act_v)  
        # q,k,v: [B, H, l2, D]

        q = torch.cat([img_q, act_q], dim=2)  # [B, H, l1+l2, D]
        k = torch.cat([img_k, act_k], dim=2)  # [B, H, l1+l2, D]
        v = torch.cat([img_v, act_v], dim=2)  # [B, H, l1+l2, D]
        attn = F.scaled_dot_product_attention(q, k, v)  # joint attention between img and act

        # cross atten for tactile
        if self.use_tactile and tactile is not None:
            tactile_k = self.tactile_key(tactile)  # [B, 64, dim]
            tactile_v = self.tactile_value(tactile)  # [B, 64, dim]
            tactile_k = einops.rearrange(tactile_k, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)
            tactile_v = einops.rearrange(tactile_v, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)

            cross_attn = F.scaled_dot_product_attention(q, tactile_k, tactile_v)  # joint attention with tactile
            attn = attn + cross_attn  # combine attention outputs
        
        attn = einops.rearrange(attn, "B H L D -> B L (H D)")
        img_attn, act_attn = attn[:, :total_img_len, :], attn[:, total_img_len:, :]  # split img and action

        img = img + gate1_feat * self.img_proj(img_attn) # residual after attn proj layer
        if self.use_tactile and self.plugin:
            img = img + gate1_feat * self.img_proj_pi(img_attn)

        # 与 DECO 原生源码保持一致：PI_Adapter MLP 在主 MLP residual 写回后再次读取 img/act，
        # 形成串行 residual 叠加，而不是共享同一个 pre-norm 输入的并行分支。
        img = img + gate2_feat * self.img_mlp((1 + scale2_feat) * self.img_norm2(img) + shift2_feat) # residual after mlp layer
        if self.use_tactile and self.plugin:
            img = img + gate2_feat * self.img_mlp_pi((1 + scale2_feat) * self.img_norm2(img) + shift2_feat)


        act = act + gate1_act * self.act_proj(act_attn)  # residual after attn proj layer
        if self.use_tactile and self.plugin:
            act = act + gate1_act * self.act_proj_pi(act_attn)

        act = act + gate2_act * self.act_mlp((1 + scale2_act) * self.act_norm2(act) + shift2_act)  # residual after mlp layer
        if self.use_tactile and self.plugin:
            act = act + gate2_act * self.act_mlp_pi((1 + scale2_act) * self.act_norm2(act) + shift2_act)

        return img, act


class adaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim,  dim * 6)
        self.silu = nn.SiLU()

    def forward(self, vec: Tensor):
        out = self.linear(self.silu(vec))
        scale1, shift1, gate1, scale2, shift2, gate2 = out[:, None, :].chunk(6, dim=-1)
        return scale1, shift1, gate1, scale2, shift2, gate2

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class timeEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

def modeling(
    action_dim,
    chunk_size,
    obs_state,
    use_tactile=False,
    plugin=False,
    plugin_rank=32,
    use_task_condition=False,
    num_tasks=10,
    inf_step=10,
    num_attn_blocks=6,
    heads=8,
    dim=512,
    rope_axes_dim=(256, 256),
    vision_backbone="resnet34",
    depth_backbone="resnet34",
):
    return DECO(
        act_dim=action_dim,
        chunk_size=chunk_size,
        obs_state=obs_state,
        use_tactile=use_tactile,
        plugin=plugin,
        plugin_rank=plugin_rank,
        use_task_condition=use_task_condition,
        num_tasks=num_tasks,
        inf_step=inf_step,
        num_attn_blocks=num_attn_blocks,
        heads=heads,
        dim=dim,
        rope_axes_dim=rope_axes_dim,
        vision_backbone=vision_backbone,
        depth_backbone=depth_backbone,
    )
