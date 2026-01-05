import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from config import Config

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete

from rlbench.tasks import PickUpCup, PickAndLift, StackBlocks


def choose_task(name: str):
    name = name.lower()
    if name == "pickupcup":
        return PickUpCup
    if name == "pickandlift":
        return PickAndLift
    if name == "stackblocks":
        return StackBlocks
    raise ValueError("TASK_NAME must be one of: pickupcup / pickandlift / stackblocks")


def make_obs_config(image_key: str) -> ObservationConfig:
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

    cam.rgb = True
    obs_cfg.gripper_pose = True
    obs_cfg.gripper_open = True
    return obs_cfg


def resize_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def occlude_center(img: np.ndarray, half: int) -> np.ndarray:
    out = img.copy()
    h, w, _ = out.shape
    cx, cy = w // 2, h // 2
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    out[y0:y1, x0:x1] = 0
    return out


# Quaternion utilities (x,y,z,w)
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=np.float32)


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = aa.astype(np.float32)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = aa / angle
    half = 0.5 * angle
    s = np.sin(half)
    x, y, z = axis * s
    w = np.cos(half)
    return quat_normalize(np.array([x, y, z, w], dtype=np.float32))


def clip_axis_angle(aa: np.ndarray, max_angle: float) -> np.ndarray:
    angle = float(np.linalg.norm(aa))
    if angle <= max_angle:
        return aa.astype(np.float32)
    if angle < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return (aa * (max_angle / angle)).astype(np.float32)


class DeltaPoseBC(nn.Module):
    def __init__(self, out_dim: int, state_dim: int, use_state: bool, use_pretrained: bool):
        super().__init__()
        self.use_state = use_state

        if use_pretrained:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        if self.use_state:
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            head_in = 512 + 64
        else:
            head_in = 512

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, state=None):
        img_feat = self.encoder(x)
        img_feat = torch.flatten(img_feat, 1)

        if self.use_state:
            if state is None:
                raise ValueError("State input is required when use_state=True")
            state_feat = self.state_mlp(state)
            feat = torch.cat([img_feat, state_feat], dim=1)
        else:
            feat = img_feat

        return self.head(feat)


def main():
    cfg = Config()
    task_name = os.environ.get("TASK_NAME", "pickupcup")

    ckpt_path = cfg.work_dir / f"deltaposebc_{task_name}" / "ckpt_last.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}. Run train_bc.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[eval] device:", device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", {})
    target_dim = int(ckpt_cfg.get("target_dim", cfg.target_dim))
    use_state = bool(ckpt_cfg.get("use_state", cfg.use_state))
    state_dim = int(ckpt_cfg.get("state_dim", cfg.state_dim))
    normalize_state = bool(ckpt_cfg.get("normalize_state", cfg.normalize_state))
    demo_stride = max(int(ckpt_cfg.get("demo_stride", cfg.demo_stride)), 1)
    image_key = ckpt_cfg.get("image_key", cfg.image_key)
    img_w = int(ckpt_cfg.get("img_w", cfg.img_w))
    img_h = int(ckpt_cfg.get("img_h", cfg.img_h))
    max_delta_xyz = float(ckpt_cfg.get("max_delta_xyz", cfg.max_delta_xyz))
    max_delta_angle = float(ckpt_cfg.get("max_delta_angle", cfg.max_delta_angle))
    gripper_threshold = float(ckpt_cfg.get("gripper_threshold", cfg.gripper_threshold))

    state_mean = ckpt.get("state_mean")
    state_std = ckpt.get("state_std")
    if state_mean is not None:
        state_mean = np.array(state_mean, dtype=np.float32)
    if state_std is not None:
        state_std = np.array(state_std, dtype=np.float32)
    if not normalize_state:
        state_mean = None
        state_std = None

    model = DeltaPoseBC(
        out_dim=target_dim,
        state_dim=state_dim,
        use_state=use_state,
        use_pretrained=cfg.use_pretrained,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    action_mode = ActionMode(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    obs_config = make_obs_config(image_key)

    env = Environment(action_mode, obs_config=obs_config, headless=cfg.headless)
    env.launch()

    task_cls = choose_task(task_name)
    task = env.get_task(task_cls)

    success = 0
    total = cfg.num_eval_episodes

    for _ in tqdm(range(total), desc=f"Eval {task_name} (IK, delta) occ={cfg.occlusion}"):
        _, obs = task.reset()
        reward = 0.0
        done = False

        for _t in range(cfg.max_steps):
            img = getattr(obs, image_key)
            img = resize_rgb(img, img_w, img_h)
            if cfg.occlusion:
                img = occlude_center(img, cfg.occ_half)

            x = tf(img).unsqueeze(0).to(device, non_blocking=True)

            pose7 = np.array(obs.gripper_pose, dtype=np.float32)  # [x,y,z,qx,qy,qz,qw]
            curr_xyz = pose7[:3]
            curr_q = quat_normalize(pose7[3:7])

            if use_state:
                curr_open = float(obs.gripper_open)
                state = np.concatenate([pose7, np.array([curr_open], dtype=np.float32)], axis=0)
                if state_mean is not None:
                    state = (state - state_mean) / state_std
                s = torch.from_numpy(state).unsqueeze(0).to(device, non_blocking=True)
            else:
                s = None

            with torch.no_grad():
                pred = model(x, s).squeeze(0)  # (7,)

            pred = pred.detach().cpu().numpy().astype(np.float32)

            # Parse outputs
            pred_delta = np.tanh(pred[:6]).astype(np.float32)
            delta_xyz = pred_delta[:3] * max_delta_xyz
            delta_aa = pred_delta[3:6] * max_delta_angle
            if demo_stride > 1:
                delta_xyz = delta_xyz / float(demo_stride)
                delta_aa = delta_aa / float(demo_stride)
            g_logit = pred[6]

            # Constrain step sizes
            delta_xyz = np.clip(delta_xyz, -max_delta_xyz, max_delta_xyz).astype(np.float32)
            delta_aa = clip_axis_angle(delta_aa, max_delta_angle)

            # Compose target pose
            tgt_xyz = (curr_xyz + delta_xyz).astype(np.float32)
            dq = axis_angle_to_quat(delta_aa)
            tgt_q = quat_normalize(quat_mul(dq, curr_q))

            # Gripper command (Discrete)
            g_prob = 1.0 / (1.0 + np.exp(-float(g_logit)))
            g_cmd = 1.0 if g_prob >= gripper_threshold else 0.0

            action = np.concatenate([tgt_xyz, tgt_q, np.array([g_cmd], dtype=np.float32)], axis=0).astype(np.float32)
            obs, reward, done = task.step(action)
            if done:
                break

        if reward > 0.0:
            success += 1

    env.shutdown()
    print(f"[eval] task={task_name} occ={cfg.occlusion} success={success}/{total}={success/total:.3f}")


if __name__ == "__main__":
    main()
