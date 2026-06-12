# DECO Action Dispatch Implementation Plan

> **For agentic workers:** 按 checkbox 顺序执行；本仓库只允许静态修改与静态审查，不在本机运行 Python、MuJoCo、ROS 或测试。

**Goal:** 将 Kuavo-DECO 部署默认行为改为 30Hz Receding Horizon，同时提供 Temporal Ensembling 与显式 Stride Action，并生成可用于动作抽搐排查的时间诊断文件。

**Architecture:** 模型继续输出完整 action chunk。`CustomDECOPolicyWrapper` 将 chunk 交给唯一 dispatcher；部署配置负责模式选择和时间语义校验，环境负责记录 clipping 与真实指令发送时间，eval 入口负责批量写出诊断结果。

**Branch:** `deco/fix/action-state` -> `deco/dev` -> `deco/main`

---

## Task 1: Action Dispatcher

- [x] 新增统一 dispatcher 接口及 Receding Horizon、Temporal Ensembling、Stride Action 三种互斥实现。
- [x] 默认创建 Receding Horizon；`n_action_steps=null` 完整消费原始 chunk。
- [x] wrapper 的 `select_action()` 只委托 dispatcher，`reset()` 清空全部跨周期状态。
- [x] 保持 dispatcher 位于 postprocessor 前，不修改训练 forward、loss 或权重。

## Task 2: Deploy Configuration

- [x] 在 deploy YAML 增加 `deco.action_dispatch` 和 `deco.timing_diagnostics`。
- [x] 默认 `mode=receding_horizon`、`env.ros_rate=30`。
- [x] 加载 checkpoint 后校验 dataset Hz、环境 Hz、chunk size、N 步和 stride 整除关系。
- [x] 本地仿真、真机和 server 使用统一配置入口；server 保存 dispatcher 状态。
- [x] 旧 deploy YAML 缺少新字段时迁移到 Receding Horizon，并输出提示，不再隐式启动 Stride。
- [x] 修复 `deco.inf_step` 部署接口：`null`/缺失沿用 checkpoint，正整数同步覆盖 `policy.config.inf_step` 与 `policy.model.inference_step`。
- [x] 本地仿真、真机和 server 统一通过 `configure_deco_runtime()` 应用推理步数与 dispatcher，并记录最终值和来源。

## Task 3: Timing Diagnostics

- [x] 记录 preprocess、policy call、真实模型推理、postprocess、env step、ROS sleep、观测读取和完整控制周期。
- [x] 记录 chunk ID、动作索引、队列长度、ensemble count、clipping 前后动作及 arm/eef 指令发送时间。
- [x] 批量写出 `deco_timing_trace.jsonl` 和 `deco_timing_summary.json`。
- [x] Summary 计算 P50/P95/P99、实际命令 Hz、deadline miss、clip 比例和 chunk 边界耗时。

## Task 4: Static Tests And Documentation

- [x] 添加 dispatcher 与部署配置测试，覆盖默认模式、连续索引、stride 索引、ensemble 公式、reset 和非法配置。
- [x] 更新 `PLANS.md`、技术决策、deploy/policy YAML 和 `AI_Logs.md`。
- [ ] 在允许运行的环境中执行单元测试。
- [ ] 在 MuJoCo 中按 Receding Horizon、Temporal Ensembling、Stride baseline 顺序完成对照验证。

## Acceptance Criteria

- [x] 静态代码中默认 dispatcher 为 Receding Horizon，checkpoint 的 `action_stride` 不会隐式生效。
- [x] 三种模式具有独立参数和互斥校验。
- [x] 部署时间诊断能够区分队列动作与新模型推理动作。
- [x] `deco.inf_step` 不再触发未知配置字段错误，且不会改变模型权重、chunk size 或动作执行步数。
- [ ] 30Hz Receding Horizon 的实际指令周期 P95 接近 33.3ms；未达到时必须根据诊断报告记录实时性失败。
