# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run kuavo_train/train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from lerobot_patches import custom_patches

from pathlib import Path

from sympy import im
from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.policy.deco.DECOPolicyWrapper import CustomDECOPolicyWrapper
from kuavo_train.wrapper.policy.deco import DECOProcessor  # noqa: F401 - 注册 DECO preprocessor step
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.random_utils import set_seed
import datetime
import time
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading

from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.utils.deco_timing import DecoTimingRecorder
from kuavo_deploy.kuavo_service.client import PolicyClient
from kuavo_deploy.utils.deco_obs_action import configure_deco_action_dispatch, validate_deco_policy_compatibility
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.policies.factory import make_pre_post_processors

log_model = setup_logger("model")
log_robot = setup_logger("robot")

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)
stop_flag = threading.Event()
pause_flag = threading.Event()


def setup_policy(pretrained_path, policy_type, device=torch.device("cuda"), inference_config=None):
    """
    Set up and load the policy model.
    
    Args:
        pretrained_path: Path to the checkpoint
        policy_type: Type of policy ('diffusion', 'act', 'deco', or 'client')
        inference_config: Optional ConfigInference for client host/port/token settings.
        
    Returns:
        Loaded policy model and device
    """
    
    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)  
    
    policy_type_key = policy_type.lower()
    if policy_type_key == 'diffusion':
        policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type_key == 'act':
        policy = CustomACTPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type_key == 'deco':
        policy = CustomDECOPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type_key == 'client':
        if inference_config is None:
            policy = PolicyClient()
        else:
            # token 只从环境变量读取，避免把远端推理服务凭据写入 YAML 或日志。
            api_token = inference_config.client_api_token_value()
            policy = PolicyClient(
                host=inference_config.client_host,
                port=inference_config.client_port,
                timeout_ms=inference_config.client_timeout_ms,
                api_token=api_token,
            )
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")

    if policy_type_key != 'client':
        policy.eval()
        policy.to(device)
    if hasattr(policy, "reset"):
        policy.reset()
    # Log model info
    log_model.info(f"Model loaded from {pretrained_path}")
    if hasattr(policy, "config"):
        log_model.info(f"Model n_obs_steps: {getattr(policy.config, 'n_obs_steps', 'N/A')}")
    log_model.info(f"Model device: {device}")
    
    return policy

def main(config: KuavoConfig, env: gym.Env):
    # load config
    cfg = config.inference

    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    start_seed = cfg.start_seed
    policy_type = cfg.policy_type
    task = cfg.task
    method = cfg.method
    timestamp = cfg.timestamp
    epoch = cfg.epoch
    env_name = cfg.env_name

    pretrained_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")
    output_directory = Path(f"outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}")
    # Create a directory to store the video of the evaluation
    output_directory.mkdir(parents=True, exist_ok=True)
    timing_config = config.deco.timing_diagnostics
    timing_recorder = DecoTimingRecorder(
        output_directory,
        enabled=timing_config.enabled and config.env.state_layout.startswith("deco_"),
        flush_every_steps=timing_config.flush_every_steps,
        target_hz=config.env.ros_rate,
    )

    # set seed
    set_seed(seed=seed)

    # Select your device
    device = torch.device(cfg.device)

    policy = setup_policy(pretrained_path, policy_type, device, cfg)
    if policy_type.lower() == 'deco':
        validate_deco_policy_compatibility(policy.config, config.deco, config.env)
        configure_deco_action_dispatch(policy, config.deco, config.env)
        log_model.info(f"DECO action dispatch: {policy.get_dispatch_info()}")
    # preprocessor = PolicyProcessorPipeline.from_pretrained(pretrained_path, config_filename="policy_preprocessor.json")
    # postprocessor = PolicyProcessorPipeline.from_pretrained(pretrained_path, config_filename="policy_postprocessor.json")
    preprocessor, postprocessor = make_pre_post_processors(None, Path(str(pretrained_path).split("/epoch", 1)[0]))

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent.
    max_episode_steps = cfg.max_episode_steps
    env = env

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    if policy_type.lower() != 'client':
        log_model.info(f"policy.config.input_features: {policy.config.input_features}")
        log_robot.info(f"env.observation_space: {env.observation_space}")

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    if policy_type.lower() != 'client':
        log_model.info(f"policy.config.output_features: {policy.config.output_features}")
        log_robot.info(f"env.action_space: {env.action_space}")

    # Log evaluation results
    log_file_path = output_directory / "evaluation.log"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")

    success_count = 0
    for episode in tqdm(range(eval_episodes), desc="Evaluating model", unit="episode"):
        # Reset the policy and environments to prepare for rollout
        policy.reset()
        observation, info = env.reset(seed=episode+start_seed)
        observation = preprocessor(observation)
        # log_file.write(f"~~~~~~~~~~~~~~~~~~preprocess observation ok!~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []

        cam_keys = [k for k in observation.keys() if "images" in k or "depth" in k]
        frame_map = {k: [] for k in cam_keys}

        average_exec_time = 0
        average_action_infer_time = 0
        average_step_time = 0

        step = 0
        done = False
        with tqdm(total=max_episode_steps, desc=f"Episode {episode+1}", unit="step", leave=False) as pbar:
            while not done:
                # --- Pause support: block here if pause_flag is set ---
                while pause_flag.is_set() and not stop_flag.is_set():
                    log_model.info("Paused. Waiting for resume signal...")
                    time.sleep(0.5)
                if stop_flag.is_set():
                    log_model.info("Stop flag detected during pause. Exiting loop.")
                    return
                
                control_loop_start_ns = time.perf_counter_ns()
                start_time = time.time()

                policy_call_start_ns = time.perf_counter_ns()
                with torch.inference_mode():
                    action = policy.select_action(observation)
                policy_call_end_ns = time.perf_counter_ns()
                raw_policy_action = action.detach().cpu().tolist()
                postprocess_start_ns = time.perf_counter_ns()
                action = postprocessor(action)
                postprocess_end_ns = time.perf_counter_ns()
                action_infer_time = time.time()
                log_model.debug(f"action infer time: {action_infer_time - start_time:.3f}s")
                average_action_infer_time += action_infer_time - start_time

                numpy_action = action.squeeze(0).cpu().numpy()
                log_model.debug(f"numpy_action: {numpy_action}")

                # 执行动作 Execute action
                environment_step_start_ns = time.perf_counter_ns()
                observation, reward, terminated, truncated, info = env.step(numpy_action)
                environment_step_end_ns = time.perf_counter_ns()
                preprocess_start_ns = time.perf_counter_ns()
                observation = preprocessor(observation)
                preprocess_end_ns = time.perf_counter_ns()
                exec_time = time.time()
                log_model.debug(f"exec time: {exec_time - action_infer_time:.3f}s")
                average_exec_time += exec_time - action_infer_time

                rewards.append(reward)

                control_loop_end_ns = time.perf_counter_ns()
                dispatch_info = policy.get_dispatch_info() if hasattr(policy, "get_dispatch_info") else {}
                raw_env = getattr(env, "unwrapped", env)
                env_diagnostics = getattr(raw_env, "last_step_diagnostics", {})
                timing_recorder.record(
                    {
                        "episode": episode,
                        "step": step,
                        "mode": dispatch_info.get("mode", config.deco.action_dispatch.mode),
                        "chunk_id": dispatch_info.get("chunk_id"),
                        "action_index": dispatch_info.get("action_index"),
                        "queue_remaining": dispatch_info.get("queue_remaining"),
                        "model_inference": dispatch_info.get("model_inference"),
                        "ensemble_count": dispatch_info.get("ensemble_count"),
                        "preprocess_time_ns": preprocess_end_ns - preprocess_start_ns,
                        "policy_call_time_ns": policy_call_end_ns - policy_call_start_ns,
                        "model_inference_time_ns": dispatch_info.get("inference_time_ns"),
                        "postprocess_time_ns": postprocess_end_ns - postprocess_start_ns,
                        "environment_step_time_ns": environment_step_end_ns - environment_step_start_ns,
                        "control_loop_time_ns": control_loop_end_ns - control_loop_start_ns,
                        "raw_policy_action": raw_policy_action,
                        "postprocessed_action": numpy_action.tolist(),
                        **env_diagnostics,
                    }
                )

                # 相机帧记录，真机请取消，否则会一直堆叠卡死
                # Camera frame record, must be commented out during real-device testing

                # for k in cam_keys:
                #     frame_map[k].append(observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0))

                # The rollout is considered done when the success state is reached (i.e. terminated is True),
                # or the maximum number of iterations is reached (i.e. truncated is True)
                done = terminated | truncated | done
                step += 1

                end_time = time.time()
                log_model.info(f"Step {step} time: {end_time - start_time:.3f}s")
                
                # Update progress bar
                status = "Success" if terminated else "Running"
                pbar.set_postfix({
                    "Reward": f"{reward:.3f}",
                    "Status": status,
                    "Total Reward": f"{sum(rewards):.3f}"
                })
                pbar.update(1)

        if terminated:
            success_count += 1
            log_model.info(f"✅ Episode {episode+1}: Success! Total reward: {sum(rewards):.3f}")
        else:
            log_model.info(f"❌ Episode {episode+1}: Failed! Total reward: {sum(rewards):.3f}")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.ros_rate

        log_model.info(f"average exec time: {average_exec_time / step:.3f}s")
        log_model.info(f"average action infer time: {average_action_infer_time / step:.3f}s")
        log_model.info(f"average step time: {average_step_time / step:.3f}s")
        log_model.info(f"average sleep time: {env.average_sleep_time / step:.3f}s")
        
        
        # Encode all frames into a mp4 video.
        if len(frame_map.keys()) == 0:
            for cam in cam_keys:
                frames = frame_map[cam]
                output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
                imageio.mimsave(str(output_path), frames, fps=fps)

        # print(f"Video of the evaluation is available in '{video_path}'.")

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Rewards per Episode: {numpy.array(rewards).sum()}")

    with log_file_path.open("a") as log_file:
        log_file.write("\n")
        log_file.write(f"Success Count: {success_count}\n")
        log_file.write(f"Success Rate: {success_count / eval_episodes:.2f}\n")

    # Display final statistics
    log_model.info("\n" + "="*50)
    log_model.info(f"🎯 Evaluation completed!")
    log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
    log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    print(f"📁 Videos and logs saved to: {output_directory}")
    print("="*50)
    timing_recorder.close()

def kuavo_eval(config: KuavoConfig, env: gym.Env):
    main(config, env)

if __name__ == "__main__":
    config_path = Path("test.yaml")
    env = gym.make(
        "Kuavo-Real",
        max_episode_steps=150,
        config_path=config_path,
    )
