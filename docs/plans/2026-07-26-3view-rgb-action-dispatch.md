# 3View RGB Action Dispatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** 在 3View RGB 分支移植 `deco/fix/action-state` 的三模式动作后端，并将默认行为固定为 32-step chunk 连续执行前16步。

**Architecture:** 动作模型仍输出 `[1,32,action_dim]`；独立 dispatcher 只负责在线时间消费方式。默认 Receding Horizon 在30Hz下执行索引 `0..15`，队列耗尽后使用最新观测重新推理；Temporal Ensembling 与 Stride Action 作为互斥可选模式保留。

**Tech Stack:** Python、PyTorch、LeRobot policy wrapper、Kuavo ROS/仿真/server-client 部署配置。

---

### Task 1: 移植动作分发模块

**Files:**
- Create: `kuavo_train/wrapper/policy/deco/action_dispatch.py`
- Modify: `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py`

- [x] 移植 Receding Horizon、Temporal Ensembling 和 Stride Action。
- [x] wrapper 的 `select_action()` 只委托唯一 dispatcher。
- [x] 默认 Receding Horizon 使用 `n_action_steps=16`。

### Task 2: 固定配置与时间语义

**Files:**
- Modify: `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py`
- Modify: `configs/policy/deco_config.yaml`
- Modify: `kuavo_deploy/config.py`
- Modify: `configs/deploy/kuavo_deco_env.yaml`

- [x] 删除活动 policy 配置中的 `control_hz/action_stride`。
- [x] 部署 YAML 暴露三种互斥模式。
- [x] Receding Horizon 与 Temporal Ensembling 强制 `env.ros_rate == dataset_hz`。
- [x] 默认 `dataset_hz=30`、`env.ros_rate=30`、`n_action_steps=16`。
- [x] `deco.inf_step=null` 保持 checkpoint 值，正整数同时覆盖 policy config 与模型 denoising 循环。

### Task 3: 覆盖全部部署入口

**Files:**
- Modify: `kuavo_deploy/utils/deco_obs_action.py`
- Modify: `kuavo_deploy/src/eval/real_single_test.py`
- Modify: `kuavo_deploy/src/eval/sim_auto_test.py`
- Modify: `kuavo_deploy/kuavo_service/server.py`

- [x] 实机、仿真与 server policy 侧使用同一 dispatcher 配置。
- [x] server/client 模式的队列状态保留在 server policy 侧。
- [x] 保留 3View RGB 三相机兼容性检查。

### Task 4: 验收

**Files:**
- Create: `tests/test_deco_action_dispatch.py`
- Create: `tests/test_deco_deploy_action_dispatch_config.py`
- Modify: `PLANS.md`
- Modify: `AI_Logs.md`

- [x] 静态覆盖16步连续动作、第17步重新推理、reset、三模式及频率约束。
- [ ] 在允许执行代码的环境运行测试、仿真和实机30Hz周期验证。
