# Kuavo 数据挑战赛：全局工具链架构与运行逻辑

本文档梳理了 `kuavo_data_challenge` 仓库从数据收集到真机部署的完整端到端链路。了解此链路有助于为后续接入定制化模型（如 DECO）提供清晰的架构认知。

---

## 1. 数据收集与清洗转换 (`kuavo_data`)

这是整个链路的起点。

- **背景**：真实世界的 Kuavo 机器人（或 MuJoCo 仿真环境）在运行和遥操作时，会将所有的传感器数据、多相机图像、关节角状态等以 ROS 的专属格式（`.bag` 文件，即 **rosbag**）记录下来。
- **本环节任务**：底层的深度学习框架（无论是 LeRobot 还是未来的 DECO）无法直接高效读取 rosbag。因此，必须使用 `kuavo_data/CvtRosbag2Lerobot.py` 脚本负责“清洗和转换”。
- **工作流**：该脚本解析 rosbag，提取出需要的 RGB 图像、深度图、关节动作、触觉数据等，并将它们按时间戳严格对齐（如 30Hz 抽帧），最终打包存储为 **Parquet 格式**。这是 Hugging Face 数据集的一种高效、标准的存储格式。

## 2. 第三方核心引擎 (`third_party/lerobot`)

- **组件定位**：存放在 `third_party` 目录下，是 Hugging Face 开源的机器人具身智能核心框架 **LeRobot**（作为 Git Submodule 引入）。
- **作用**：可以将其理解为 PyTorch 生态里的“通用发动机”。它自带了极其完善且优化的逻辑：
  - Dataset 读取（直接对接上一步生成的 Parquet 格式）。
  - Dataloader 数据并行加载与预处理增强。
  - 标准化的训练循环 (Train Loop) 与优化器调度。
  - 标准策略的基线实现（如原版的 ACT 和 Diffusion）。
- **与当前项目的关系**：Kuavo 项目的训练模块（`kuavo_train`）是“构建”在 LeRobot 之上的。我们调用它的上层接口启动训练，从而免去了从头编写大量底层深度学习循环代码的工作。

## 3. 为什么需要 Wrapper？（适配与定制层）

这是接入任何第三方定制模型（如 DECO）的**主战场**。

- **动机**：LeRobot 是一个通用的开源库，期望接收标准化的输入（如单一的图像键值、统一的状态向量）。然而，Kuavo 拥有特殊的硬件架构（如左右手多个深度相机，未来还会加入高维触觉传感器）。如果直接修改 `third_party/lerobot` 的代码，会破坏开源库的完整性，导致难以合并官方的上游更新。
- **Wrapper 的本质**：它是一个设计模式上的 **“翻译官”** 或 **“适配器”**，由以下核心部分组成：
  - **ConfigWrapper (`*ConfigWrapper.py`)**：负责把 Kuavo 专属的超参数（如是否开启深度融合、特殊网络层数）与 LeRobot 的配置系统对接。
  - **ModelWrapper (`*ModelWrapper.py`)**：拦截 LeRobot 数据加载器传来的标准数据字典（`batch`），在内部解包并执行 Kuavo 特有的融合逻辑（如 RGB 和 Depth 的 Cross-Modal Fusion），最后将处理好的张量传递给核心的神经网络（如 ACT/Diffusion 或 DECO）。
  - **PolicyWrapper (`*PolicyWrapper.py`)**：负责计算模型的损失函数 (Loss)。LeRobot 的引擎期望得到一个 `loss` 标量，因此各种模型特有的损失计算（如 Diffusion 的加噪去噪 Loss）必须封装在这一层返回。
- **动态补丁 (`lerobot_patches`)**：这是一种特殊形式的适配（Monkey Patch）。在 Python 启动时，动态注入或修改 LeRobot 的部分基础类（如扩展 `FeatureType` 以支持 Depth 统计），从而在不改动源码的前提下实现底层功能扩展。

## 4. 训练与部署闭环 (`kuavo_train` & `kuavo_deploy`)

- **模型训练 (`kuavo_train/train_policy.py`)**：这是一个非常轻量级的启动脚本。它负责读取 `configs/policy` 中的 YAML 参数，实例化带有 Wrapper 的策略模型，然后将控制权完全移交给 LeRobot 的 Trainer。训练过程中会不断在 `outputs/` 下生成 `.safetensors` 模型权重。
- **模型部署 (`kuavo_deploy`)**：
  - **仿真端**：加载训练好的模型权重，通过驱动 MuJoCo 仿真器中的数字孪生 Kuavo，进行闭环或开环的虚拟环境测试。
  - **真机端**：加载模型并启动对应的 ROS 推理节点，将神经网络输出的 Action 直接发布给真机的底层硬件驱动（如灵巧手、夹爪、机械臂等）。

---

## 💡 总结：对齐 DECO 的大局思路

基于上述架构，将 DECO 接入 Kuavo 工具链的宏观路径如下：

1. **数据层适配 (`kuavo_data`)**：在 `CvtRosbag2Lerobot.py` 附近，确保 Kuavo 的触觉数组被正确提取，并同图像数据一起写入 Parquet 数据集。
2. **配置层扩展 (`configs`)**：新建 `configs/policy/deco_config.yaml`，配置诸如 `RGB_Augmenter` 等数据增强策略及 DECO 网络超参数。
3. **编写适配器 (`kuavo_train/wrapper/policy/deco`)**：
   - 编写 `DECOModelWrapper`：拆解 Parquet 读出的 `batch` 字典，将特定的触觉与图像张量喂给 DECO 原生的 `forward` 函数。
   - 编写 `DECOPolicyWrapper`：在其中实现并计算 DECO 的 Diffusion Loss，返回给外层引擎。
4. **架构守护**：确保整个过程**不修改 DECO 核心源码**、**不修改 `train.py`**，且**不破坏现有的 ACT 和 DP 基线**。
