# DECO 集成 - 详细实施计划书 (Implementation Plan)

本计划书细化了各阶段的具体代码落地逻辑。我们将逐步补全所有阶段，在您统一 Review 后再开始写码。

---

## 阶段一：数据引擎阶段 (Data Engine Phase)
*状态：逻辑已对齐，等待最终执行*

### 1. `kuavo_data/common/kuavo_dataset.py`
1. **触觉特征提取**：
   - 新增 `process_touch_state` 方法。提取左右手的 `normal_force1/2/3`，构成 30 维张量。
   - **数值量纲处理**：基于您的要求，直接除以 100，将其从 0~2500 缩放至 **0~25 牛顿** 的物理区间。
2. **频率对齐**：
   - 依赖现有 `align_frame_data` 中 `np.argmin(np.abs(time_array - stamp_sec))` 实现 100Hz 到 30Hz 的最近邻降采样。

### 2. `kuavo_data/CvtRosbag2Lerobot_DECO.py` (新文件)
1. **动作空间 (28 维) 映射**：
   - 新增 `KuavoDecoMapper`，将 `臂(0-6, 7-13) -> 手(14-19, 20-25) -> 头(26-27)` 重组为 `左臂(0-6) -> 左手(7-12) -> 右臂(13-19) -> 右手(20-25) -> 头部(26-27)`。
   - **头部缺失处理**：对于未录制头部话题的强校验，直接对 Index 26-27 **补全 0.0**。
2. **特征注册**：
   - `create_empty_dataset` 中注册 `observation.tactile`，shape 为 `(30,)`。

---

## 阶段二：工具链整合阶段 (Toolchain Integration Phase)
*状态：逻辑已对齐，采用方案 1 动态注入法*

### 1. 代码物理迁移
- 将根目录下的 `DECO/` 文件夹整体移动（重命名）为 `third_party/deco/`。

### 2. Python 依赖剥离与合并
- 读取现有的 `DECO/requirements.txt`。分离出特有依赖库（如 `timm`, `einops`）并新建项目根目录的 `requirements_DECO.txt`。

### 3. 包路径引用桥接 (Path Bridging)
- 在未来调用的包裹层顶部，添加 `sys.path.append("third_party/deco")`，零侵入解决 `from models...` 的报错。

---

## 阶段三：硬件适配与模型验证阶段 (Model Surgery Phase)
*状态：草案设计中*

### User Review Required
> [!IMPORTANT]
> **关于网络维度的手术（请查阅并在下方确认）**：
> DECO 原始网络是给 Inspire Hand（拥有 1062 维触觉，前向传播时会求平均降成 17 维特征）设计的。
> 我们现在传入的是精确的 30 维（左手 15 + 右手 15）`normal_force` 特征，因此我计划**直接删掉它的降维均值操作**，让这 30 维原封不动地进入融合网络。
> 我们需要改动底层网络结构的 Linear 的 input shape：原先 `17+17+34 = 68` 维，现在会变成 `15+15+34 = 64` 维。
> **请确认**：您是否同意这种将底层模型 Linear in_features 改为匹配我们新维度（64 维）的“硬截肢”操作？这将破坏它原本加载基于 Inspire Hand 的 pretrained weights 的能力（如果您原本打算用基于触觉预训练的权重的话；如果只加载视觉预训练权重则毫无影响）。

### 1. 深度开刀点：`third_party/deco/models/deco/deco.py`

#### [MODIFY] deco.py
1. **删除 `init_tac_regions` 函数**：
   - 因为我们不再需要那 17 个 Inspire hand 的硬编码切片区域。

2. **修改 `DECO.__init__` 中的层维度**：
   - 原代码 `self.tactile_encoder = nn.Sequential(nn.Linear(1062*2, 512)...)` 
     👉 **改为** `nn.Linear(15*2, 512)` 即 30。
   - 原代码 `self.gated = nn.Linear(68, 68, bias=False)` (其中 68 = 17+17+34) 
     👉 **改为** `nn.Linear(64, 64, bias=False)` (其中 64 = 15+15+34)

3. **修改 `DECO.forward` 中的前向传递均值逻辑**：
   - 移除原来的循环求均值代码：
     ```python
     tac1_avg = torch.stack([tac1[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()], dim=1)
     ```
   - **替换为**最纯粹的透传：
     ```python
     tac1_avg = tac1  # 我们传进来的直接就是精简版的左手空间特征 (B, 15)
     tac2_avg = tac2  # 右手空间特征 (B, 15)
     ```

### 2. 静态断言验证 (Dummy Forward Pass)
- 写一个测试脚本 `test_deco_shape.py`，构造假的输入 Tensor：
  `img1/img2: (2, 3, 256, 256)`, `obs: (2, 28)`, `act: (2, 32, 28)`, `tac1/tac2: (2, 15)`。
- 走一次前向传播，确保模型能正常计算不报 Shape 错，并输出带有 `loss` 梯度的噪声残差。

---

## 阶段四：核心封装阶段 (Core Wrapper Phase)
*状态：草案设计中*

为了严格遵循“`train.py` 不可动”原则，我们需要在 `kuavo_train/wrapper/policy/deco/` 下构建 LeRobot 框架的拦截器。

### 1. `DECOPolicyWrapper.py` (训练与推理的统一入口)

#### [NEW] kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py
我们将新建这个文件，继承自 `nn.Module`，对外伪装成标准的 LeRobot Policy。

1. **头部的动态注入**：
   ```python
   import sys, os
   # 动态注入 Phase 2 确定的 third_party 路径
   deco_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../third_party/deco"))
   if deco_path not in sys.path:
       sys.path.append(deco_path)
   from models.deco.deco import DECO
   ```

2. **拦截前向传播 (`forward`)**：
   - 从 LeRobot 标准的 `batch` 字典中解包数据。
   - `img1` 和 `img2`：由于 LeRobot 给的图像是序列 `(B, T, C, H, W)`，我们需要将其切片取出当前帧和历史帧。
   - `obs`：直接对应 `batch["observation.state"]` (B, 28)。
   - **`tac1` 和 `tac2` 拆解**：第一阶段产出的 `batch["observation.tactile"]` 是完整的 (B, 30)。我们需要在传入网络前做切片：
     ```python
     tactile_full = batch["observation.tactile"]
     tac1 = tactile_full[:, :15]  # 左手 15 维
     tac2 = tactile_full[:, 15:]  # 右手 15 维
     ```
   - 调用手术后的 `DECO(training=True)`，拿到 `(act, noise)`。
   - **计算 Loss**：`loss = F.mse_loss(act, noise)`，并严格返回 `(loss, {"loss": loss.item()})`，欺骗 `train.py` 让它以为这是个原生 Policy。

3. **拦截动作预测 (`select_action`)**：
   - 解包当前观测，调用 `DECO(training=False)` 触发内部的 Flow-matching 5 步去噪循环。
   - 维护动作队列 `_queues["action"]`，每次吐出一个合法的 28 维动作供真机或仿真执行。

---

## 阶段五：配置集成阶段 (Configuration Integration Phase)
*状态：草案设计中*

### 1. 编写 `configs/policy/deco_config.yaml`

#### [NEW] configs/policy/deco_config.yaml
完全沿用 Kuavo 的 Hydra 配置规范：
1. **基础训练参数**：继承 `diffusion_config.yaml` 的标准配置（如 `batch_size: 32`, `max_epoch: 500`, `device: "cuda"`）。
2. **白嫖 Kuavo 的图像增强方案**：
   - 将 Kuavo 强大的 `RGB_Augmenter` 模块（包含 `ColorJitter`, `SharpnessJitter`, `GaussianNoise` 等）完全粘贴过来。
   - LeRobot 数据管道会自动在把图片塞进 `batch` 前完成增强，彻底解放 DECO，不需要在 DECO 代码里写任何 Data Augmentation 逻辑。
3. **注入 DECO 模型参数**：
   ```yaml
   policy_name: deco
   policy:
     _target_: kuavo_train.wrapper.policy.deco.DECOConfigWrapper.DECOConfigWrapper
     # DECO 特有超参数
     chunk_size: 32
     dim: 512
     num_attn_blocks: 6
     use_tactile: true
   ```

---

## 阶段六：部署与演示阶段 (Deployment & Demonstration Phase)
*状态：草案设计中*

### 1. `kuavo_deploy` 节点测试
确保 DECO 训练完吐出的 `.safetensors` 和 `config.json` 能够被 Kuavo 的 `eval_kuavo.py` 原生读取。
因为我们在 Phase 4 把 DECO 包装成了一个 `DiffusionPolicy` / `nn.Module`，并且对接了 `select_action`，所以部署团队的代码（`action = policy.select_action(obs_dict)`）**完全不需要知道底层是 DECO 还是 ACT**。

### 2. 闭环测试 (Loop Verification)
1. **仿真验证**：在 MuJoCo 中启动机器人，验证 28 维动作是否平滑、左右手是否没有顺拐（验证阶段一的映射逻辑是否正确）。
2. **触觉反馈验证**：在真机或仿真中施加指尖压力，观察机器人是否能通过 30 维法向力张量做出反应。
