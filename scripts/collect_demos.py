import os
import json
import random
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from config import Config

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete

from rlbench.tasks import PickUpCup, PickAndLift, StackBlocks


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def choose_task(name: str):
    name = name.lower()
    if name == "pickupcup":
        return PickUpCup
    if name == "pickandlift":
        return PickAndLift
    if name == "stackblocks":
        return StackBlocks
    raise ValueError("TASK_NAME must be one of: pickupcup / pickandlift / stackblocks")


def make_obs_config(image_key: str, use_image: bool, use_task_low_dim_state: bool) -> ObservationConfig:
    obs_cfg = ObservationConfig()
    obs_cfg.set_all(False)

    if image_key == "front_rgb":
        cam = obs_cfg.front_camera
    elif image_key == "wrist_rgb":
        cam = obs_cfg.wrist_camera
    elif image_key == "left_shoulder_rgb":
        cam = obs_cfg.left_shoulder_camera
    elif image_key == "right_shoulder_rgb":
        cam = obs_cfg.right_shoulder_camera
    elif image_key == "overhead_rgb":
        cam = obs_cfg.overhead_camera
    else:
        raise ValueError(f"Unknown image_key: {image_key}")

    if use_image:
        cam.rgb = True

    # supervision
    obs_cfg.gripper_pose = True
    obs_cfg.gripper_open = True
    if use_task_low_dim_state:
        obs_cfg.task_low_dim_state = True
    return obs_cfg


def resize_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# RLBench gripper_pose quaternion ordering is typically [x,y,z,qx,qy,qz,qw]
# We use q = [qx,qy,qz,qw]
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # (x,y,z,w) Hamilton product
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=np.float32)


def quat_inv(q: np.ndarray) -> np.ndarray:
    # for unit quaternion, inverse = conjugate
    return quat_conjugate(q)


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    # q assumed unit, (x,y,z,w). Ensure shortest path (w>=0)
    q = quat_normalize(q)
    if q[3] < 0:
        q = -q
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w*w))
    if s < 1e-6 or angle < 1e-6:
        return np.zeros((3,), dtype=np.float32)
    axis = np.array([x, y, z], dtype=np.float32) / s
    return (axis * angle).astype(np.float32)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    task_name = os.environ.get("TASK_NAME", "pickupcup")
    task_cls = choose_task(task_name)

    out_root = cfg.data_dir / task_name
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "task": task_name,
        "image_key": cfg.image_key,
        "img_w": cfg.img_w,
        "img_h": cfg.img_h,
        "num_episodes": cfg.num_episodes,
        "max_steps": cfg.max_steps,
        "demo_stride": cfg.demo_stride,
        "target_dim": cfg.target_dim,
        "use_state": cfg.use_state,
        "use_image": cfg.use_image,
        "use_task_low_dim_state": cfg.use_task_low_dim_state,
        "arm_action_mode": "EndEffectorPoseViaIK",
        "target_type": "delta_to_next_xyz + delta_to_next_axis_angle + gripper_open_next",
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    action_mode = ActionMode(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    use_task_low_dim_state = cfg.use_state and cfg.use_task_low_dim_state
    obs_config = make_obs_config(cfg.image_key, cfg.use_image, use_task_low_dim_state)

    env = Environment(action_mode, obs_config=obs_config, headless=cfg.headless)
    env.launch()
    task = env.get_task(task_cls)

    state_dim = None
    for ep in tqdm(range(cfg.num_episodes), desc=f"Collect {task_name} (IK, delta targets)"):
        descriptions, _ = task.reset()
        instruction = descriptions[0] if descriptions else ""

        demo = task.get_demos(1, live_demos=True)[0]

        ep_dir = out_root / f"ep_{ep:05d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "instruction.txt").write_text(instruction)

        stride = max(int(cfg.demo_stride), 1)
        if len(demo) <= stride:
            continue
        T = min(len(demo) - stride, cfg.max_steps)
        imgs = [] if cfg.use_image else None
        states = []
        targets = []

        for t in range(T):
            obs_t = demo[t]
            obs_tp1 = demo[t + stride]

            if cfg.use_image:
                img = getattr(obs_t, cfg.image_key)
                img = resize_rgb(img, cfg.img_w, cfg.img_h)
                imgs.append(img.astype(np.uint8))

            pose7 = np.array(obs_t.gripper_pose, dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
            next_pose7 = np.array(obs_tp1.gripper_pose, dtype=np.float32)
            curr_xyz = pose7[:3]
            curr_q = quat_normalize(pose7[3:7])
            curr_open = float(obs_t.gripper_open)
            next_xyz = next_pose7[:3]
            next_q = quat_normalize(next_pose7[3:7])
            next_open = float(obs_tp1.gripper_open)

            if cfg.use_state:
                state_parts = [pose7, np.array([curr_open], dtype=np.float32)]
                if use_task_low_dim_state:
                    low_dim = getattr(obs_t, "task_low_dim_state", None)
                    if low_dim is None:
                        raise RuntimeError("task_low_dim_state is missing from observation.")
                    low_dim = np.array(low_dim, dtype=np.float32).reshape(-1)
                    if low_dim.size == 0:
                        raise RuntimeError("task_low_dim_state is empty.")
                    state_parts.append(low_dim)
                state = np.concatenate(state_parts, axis=0).astype(np.float32)
                if state_dim is None:
                    state_dim = int(state.shape[0])
                states.append(state)

            delta_xyz = (next_xyz - curr_xyz).astype(np.float32)

            # relative rotation: q_rel = q_next * inv(q_curr)
            q_rel = quat_mul(next_q, quat_inv(curr_q))
            delta_aa = quat_to_axis_angle(q_rel)

            # target: [dx,dy,dz, ax,ay,az, gripper_open_next]
            y = np.concatenate([delta_xyz, delta_aa, np.array([next_open], dtype=np.float32)], axis=0)
            targets.append(y)

        if cfg.use_image:
            np.save(ep_dir / "images.npy", np.stack(imgs, axis=0))         # (T,H,W,3) uint8
        if cfg.use_state:
            np.save(ep_dir / "states.npy", np.stack(states, axis=0))   # (T,state_dim) float32
        np.save(ep_dir / "targets.npy", np.stack(targets, axis=0))     # (T,7) float32

    env.shutdown()
    if cfg.use_state and state_dim is not None:
        meta["state_dim"] = state_dim
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[collect] done (delta targets)")


if __name__ == "__main__":
    main()
