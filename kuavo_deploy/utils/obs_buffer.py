from dataclasses import asdict
from collections import deque
from typing import Dict, Callable, Any
import torch
import numpy as np
import cv2
import rospy
from tqdm import tqdm
import time
import sys
from kuavo_deploy.config import KuavoConfig
from sensor_msgs.msg import CompressedImage, JointState
from torchvision.transforms.functional import to_tensor
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData, lejuClawState
try:
    from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import dexhandTouchState
except ImportError:
    dexhandTouchState = None
from kuavo_deploy.utils.signal_controller import ControlSignalManager
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.utils.ros_manager import ROSManager


log_robot = setup_logger("robot")


class ObsBuffer:
    def __init__(
        self, 
        config: KuavoConfig,  
        obs_key_map: Dict[str, Dict[str, Any]] = None,
        compute_func_map: Dict[str, Callable] = None,
    ) -> None:
        self.control_signal_manager = ControlSignalManager()
        self.ros_manager = ROSManager()
        # === 从 KuavoConfig 中提取环境配置 Extract Configuration===
        # env_cfg = config.env
        env_cfg = config
        self.which_arm = env_cfg.which_arm

        # === 观测定义 Observation Defn ===
        self.obs_key_map = obs_key_map or env_cfg.obs_key_map or {}
        self.compute_func_map = compute_func_map or {}

        # === 区分订阅型与计算型观测 Differentiate Observation Types===
        self.subscribe_keys = {k: v for k, v in self.obs_key_map.items() if v.get("type") != "computed"}
        self.computed_keys  = {k: v for k, v in self.obs_key_map.items() if v.get("type") == "computed"}

        # === 反向依赖索引 Reverse Indicing Dependencies ===
        self.source_to_computed = {}
        for comp_key, comp_info in self.computed_keys.items():
            src = comp_info.get("source")
            if src:
                self.source_to_computed.setdefault(src, []).append(comp_key)
                log_robot.info(f"Registered computed obs '{comp_key}' depends on '{src}'")

        # === 初始化观测缓存 Init Observation Buffer ===
        self.obs_buffer_size = {k: v["frequency"] for k, v in self.obs_key_map.items()}
        self.obs_buffer_data = {
            k: {"data": deque(maxlen=v["frequency"]), "timestamp": deque(maxlen=v["frequency"])}
            for k, v in self.obs_key_map.items()
        }

        # === ROS topic 对应表 Reference List ===
        self.callback_key_map = {
            '/cam_h/color/image_raw/compressed': self.rgb_callback,
            '/cam_l/color/image_raw/compressed': self.rgb_callback,
            '/cam_r/color/image_raw/compressed': self.rgb_callback,
            '/cam_h/depth/image_raw/compressed': self.depth_callback,
            '/cam_h/depth/image_raw/compressedDepth': self.depth_callback,
            '/cam_l/depth/image_rect_raw/compressedDepth': self.depth_callback,
            '/cam_r/depth/image_rect_raw/compressedDepth': self.depth_callback,
            '/sensors_data_raw': self.sensorsData_callback,
            '/dexhand/state': self.qiangnaoState_callback,
            '/dexhand/touch_state': self.tactile_callback,
            '/leju_claw_state': self.lejuClawState_callback,
            '/gripper/state': self.rq2f85State_callback,
        }
        self.setup_subscribers()

    # ===== ROS订阅 Subscription =====
    def create_callback(self, callback, topic_key, handle):
        return lambda msg: callback(msg, topic_key, handle)

    def setup_subscribers(self):
        """仅订阅来自 ROS 的观测"""
        msg_type_dict = {"CompressedImage":CompressedImage,
                         "sensorsData":sensorsData,
                         "JointState":JointState,
                         "lejuClawState":lejuClawState}
        if dexhandTouchState is not None:
            msg_type_dict["dexhandTouchState"] = dexhandTouchState
        for topic_key, info in self.subscribe_keys.items():
            topic_name = info["topic"]
            assert info["msg_type"] in msg_type_dict, f"msg_type '{info['msg_type']}' is not supported; valid keys: {list(msg_type_dict.keys())}"
            msg_type = msg_type_dict[info["msg_type"]]
            callback = self.callback_key_map.get(topic_name)

            if not msg_type or not callback:
                log_robot.warning(f"Missing msg_type or callback for {topic_name}")
                continue

            handle = info.get("handle", {})
            self.ros_manager.register_subscriber(
                topic_name, msg_type, self.create_callback(callback, topic_key, handle)
            )
            log_robot.info(f"Subscribed to {topic_name} for key '{topic_key}'")

    # ===== 数据预处理 Data Preprocessing =====
    def img_preprocess(self, image):
        """图像预处理"""
        return to_tensor(image).unsqueeze(0)

    def depth_preprocess(self, depth, depth_range=[0, 1500]):
        """深度图像预处理"""
        depth_uint16 = torch.tensor(depth, dtype=torch.float32).clamp(*depth_range).unsqueeze(0)
        max_depth = depth_uint16.max()
        min_depth = depth_uint16.min()
        depth_normalized = (depth_uint16 - min_depth) / (max_depth - min_depth + 1e-9)
        return depth_normalized

    # ===== Callback 函数群 Functions =====
    def rgb_callback(self, msg: CompressedImage, key: str, handle: dict):
        img_arr = np.frombuffer(msg.data, dtype=np.uint8)
        cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Failed to decode compressed image")
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        if resize_wh:
            cv_img = cv2.resize(cv_img, resize_wh)
        data = self.img_preprocess(cv_img)
        self._append_data(key, data, msg.header.stamp.to_sec())

    def depth_callback(self, msg: CompressedImage, key: str, handle: dict):
        depth_encoding = handle.get("params", {}).get("depth_encoding", "compressedDepth_png")
        image = self._decode_depth_image(msg, depth_encoding)
        if image is None:
            return
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        if resize_wh:
            image = cv2.resize(image, resize_wh)
        if image.ndim == 3:
            # DECO 第一版只消费单通道 depth；若 ROS 封装返回多通道图像，则取第一通道保持语义稳定。
            image = image[..., 0]
        image = image[np.newaxis, ...]
        data = self.depth_preprocess(image, depth_range=handle.get("params", {}).get("depth_range", [0, 1500]))
        self._append_data(key, data, msg.header.stamp.to_sec())

    def _decode_depth_image(self, msg: CompressedImage, depth_encoding: str):
        """按配置解码 depth。

        compressed_image: 普通 sensor_msgs/CompressedImage，data 直接是可 imdecode 的图像 payload。
        compressedDepth_png: ROS compressedDepth 风格，data 前面带 header，需要定位 PNG magic。
        """
        payload = bytes(msg.data)
        if depth_encoding == "compressed_image":
            np_arr = np.frombuffer(payload, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_encoding == "compressedDepth_png":
            png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
            idx = payload.find(png_magic)
            if idx == -1:
                raise ValueError("Invalid compressedDepth message, PNG header not found")
            np_arr = np.frombuffer(payload[idx:], np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_encoding == "auto":
            image = self._decode_depth_image(msg, "compressed_image")
            if image is not None:
                return image
            return self._decode_depth_image(msg, "compressedDepth_png")
        raise ValueError(f"Unsupported depth_encoding: {depth_encoding}")

    def sensorsData_callback(self, msg: sensorsData, key: str, handle = dict):
        # Float64Array ()
        joint = msg.joint_data.joint_q
        timestamp = msg.header.stamp.to_sec()

        # FK 计算需要双臂的14个关节（索引12-26）
        # 计算依赖于此数据源的观测（例如 eef_pose）
        arm_joints = joint[12:26]  # 提取双臂关节
        self.compute_dependent_obs(key, arm_joints, timestamp)

        slice_value = handle.get("params", {}).get("slice", None)  
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, timestamp)

    def lejuClawState_callback(self, msg: lejuClawState, key: str, handle = dict):
        # Float64Array ()
        joint = msg.data.position
        slice_value = handle.get("params", {}).get("slice", None)  
        joint = [x / 100 for slc in slice_value for x in joint[slc[0]:slc[1]]] # 注意缩放
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    def qiangnaoState_callback(self, msg: JointState, key: str, handle = dict):
        joint = msg.position
        joint = [figure / 100 for figure in joint]
        slice_value = handle.get("params", {}).get("slice", None)
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    def rq2f85State_callback(self, msg: JointState, key: str, handle = dict):
        joint = msg.position
        joint = [figure / 0.8 for figure in joint]
        slice_value = handle.get("params", {}).get("slice", None)
        joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        if len(joint) == 1 and len(slice_value) == 2:
            # 与 DECO 数据转换保持一致：单值 rq2f85 视为左右夹爪对称状态。
            joint = [joint[0], joint[0]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    def tactile_callback(self, msg: Any, key: str, handle = dict):
        """解析 Kuavo 双手 tactile normal force，输出 30D 牛顿量纲向量。

        顺序与 DECO 数据转换一致：左手 5 指 × 3 normal_force，再右手 5 指 × 3 normal_force。
        """
        force_scale = float(handle.get("params", {}).get("force_scale", 100.0))
        tactile_values = []
        for hand_name in ("left_hand", "right_hand"):
            if not hasattr(msg, hand_name):
                raise ValueError(f"dexhandTouchState missing field: {hand_name}")
            fingers = list(getattr(msg, hand_name))
            if len(fingers) < 5:
                raise ValueError(f"dexhandTouchState.{hand_name} must contain at least 5 fingers.")
            for finger in fingers[:5]:
                for force_name in ("normal_force1", "normal_force2", "normal_force3"):
                    if not hasattr(finger, force_name):
                        raise ValueError(f"dexhandTouchState.{hand_name} missing field: {force_name}")
                    tactile_values.append(float(getattr(finger, force_name)) / force_scale)
        self._append_data(key, np.asarray(tactile_values, dtype=np.float32), msg.header.stamp.to_sec())

    # ===== 公共方法 Public Methods =====
    def _append_data(self, key, data, timestamp):
        self.obs_buffer_data[key]["data"].append(data)
        self.obs_buffer_data[key]["timestamp"].append(timestamp)

    def compute_dependent_obs(self, source_key, source_data, timestamp):
        for comp_key in self.source_to_computed.get(source_key, []):
            func = self.compute_func_map.get(comp_key)
            if not func:
                log_robot.warning(f"No compute function for {comp_key}")
                continue
            try:
                data = func(source_data, which_arm=self.which_arm)
                if data is not None:
                    self._append_data(comp_key, data, timestamp)
            except Exception as e:
                log_robot.error(f"Error computing {comp_key} from {source_key}: {e}")

    def obs_buffer_is_ready(self):
        return all(len(self.obs_buffer_data[k]["data"]) == self.obs_key_map[k]["frequency"] for k in self.obs_key_map)

    def stop_subscribers(self):
        self.ros_manager.close()

    def wait_buffer_ready(self):
        progress = {k: 0 for k in self.obs_key_map}
        total = {k: v["frequency"] for k, v in self.obs_key_map.items()}
        last_log_time = 0

        while not self.obs_buffer_is_ready():
            if not self.control_signal_manager.check_control_signals():
                log_robot.info("🛑 Stop signal detected, exiting")
                sys.exit(1)

            now = time.time()
            # 每隔 1 秒打印一次日志
            if now - last_log_time > 0.2:
                logs = []
                for k in progress:
                    new_len = len(self.obs_buffer_data[k]["data"])
                    progress[k] = new_len
                    logs.append(f"{k}: {new_len}/{total[k]}")
                log_robot.info(" | ".join(logs))
                last_log_time = now

            time.sleep(0.1)

        log_robot.info("✅ All buffers ready!")
        return True

    def get_latest_obs(self):
        obs = {}
        for k, buf in self.obs_buffer_data.items():
            obs[k] = list(buf["data"])[-1]  # 取最新一帧
        return obs

    def get_aligned_obs(self, reference_keys=["/cam_h/color/image_raw/compressed"], max_dt=0.01, ratio=1.0):
        """
        返回各观测时间上对齐的最新帧
        reference_keys: 以哪些key作为时间参考，默认 None -> 所有 key 最小的最新时间戳
        max_dt: 最大允许时间偏差（秒），超出则返回 None
        """
        # ===== 获取参考时间戳 =====
        if reference_keys:
            # 用指定 key 的最新时间戳
            ref_times = []
            for k in reference_keys:
                buf = self.obs_buffer_data[k]
                if len(buf) == 0:
                    continue
                _, ts = buf["data"], buf["timestamp"]
                ref_times.append(ts[-1])
            if not ref_times:
                return None
            ref_time = min(ref_times)  # 也可以取 min 或 max，根据需要
        else:
            # 没有指定 key -> 所有观测的最新 timestamp 的最小值
            last_timestamps = []
            for buf in self.obs_buffer_data.values():
                if len(buf) == 0:
                    continue
                _, ts = buf["data"], buf["timestamp"]
                last_timestamps.append(ts[-1])
            if not last_timestamps:
                return None
            ref_time = np.min(last_timestamps)

        # ===== 对齐各观测 =====
        aligned_obs = {}
        for k, buf in self.obs_buffer_data.items():
            n = int(len(buf["data"]) * ratio)
            data, ts = list(buf["data"])[-n:], list(buf["timestamp"])[-n:]
            ts = np.array(ts)
            if len(ts) == 0:
                aligned_obs[k] = None
                continue
            idx = np.argmin(np.abs(ts - ref_time))
            if abs(ts[idx] - ref_time) > max_dt:
                aligned_obs[k] = None
            else:
                aligned_obs[k] = data[idx]

        return aligned_obs

