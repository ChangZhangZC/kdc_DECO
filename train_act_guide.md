# 🚀 Kuavo ACT 训练启动指令指南

你可以直接复制下面的指令到终端运行。请根据你的实际环境修改其中的路径参数。

### 📋 核心启动指令

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=act_config.yaml \
  task=pick_and_place \
  method=act_baseline_v1 \
  root=/root/bayes-tmp/kuavo_dataset/lerobot_icra_task1/lerobot \
  training.max_epoch=100 \
  training.batch_size=16 \
  policy_name=act
```

---

### 💡 参数说明 (重要)

- **`task`**: 给你的任务起个名字（例如 `pick_and_place`）。
- **`method`**: 本次试验的标签（例如 `act_baseline_v1`），用于区分不同的训练记录。
- **`root`**: **最重要的路径！** 指向你清洗好的数据集下的 `lerobot` 文件夹（必须包含 `meta` 和 `data` 子文件夹）。
- **`training.max_epoch=100`**: 按照导师要求，设置训练 100 轮。
- **`training.batch_size=32`**: 
    - 如果训练时提示显存溢出（Out of Memory），请将其调小为 `16` 或 `8`。
    - 如果显卡性能强劲且显存充足（如 A100/H800），可以适当调大到 `64`。
- **`policy_name=act`**: 明确指定使用 ACT 模型进行训练。

---

### 🛠️ 多 GPU 加速版本 (可选)

如果你有多个 GPU，建议使用 `accelerate` 加速：

```bash
accelerate launch --config_file configs/accelerate/accelerate_config.yaml \
  kuavo_train/train_policy_with_accelerate.py \
  --config-path=../configs/policy \
  --config-name=act_config.yaml \
  task=your_task_name \
  method=act_multi_gpu \
  root=/你的/数据/存放/路径/lerobot \
  training.max_epoch=100
```

---

### 📂 产出位置
训练完成后，你的模型将保存在：
`outputs/train/<task>/<method>/run_<时间戳>/epochbest/`
