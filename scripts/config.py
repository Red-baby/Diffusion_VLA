from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    work_dir: Path = Path("/rl/rlbench_bc/runs")
    data_dir: Path = Path("/rl/rlbench_bc/data")

    headless: bool = True

    # camera: front_rgb / wrist_rgb / left_shoulder_rgb / right_shoulder_rgb / overhead_rgb
    image_key: str = "front_rgb"
    img_h: int = 128
    img_w: int = 128

    # dataset
    num_episodes: int = 10          # set 10 for smoke test; use 200+ for real runs
    max_steps: int = 180
    demo_stride: int = 1            # use t+stride as target step

    # training
    seed: int = 0
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-4
    epochs: int = 20

    # Supervised target (delta action):
    # delta_xyz (3) + delta_rot_axis_angle (3) + gripper_open (1) => 7 dims
    target_dim: int = 7
    state_dim: int = 8              # gripper_pose (7) + gripper_open (1)
    use_state: bool = True
    normalize_state: bool = True

    # action constraints (VERY IMPORTANT)
    # max translation per step (meters)
    max_delta_xyz: float = 0.04     # 4cm/step
    # max rotation per step (radians) for axis-angle magnitude
    max_delta_angle: float = 0.25   # ~14 deg/step

    # gripper: output is probability, threshold to 0/1
    gripper_threshold: float = 0.5

    # eval
    num_eval_episodes: int = 50

    # occlusion test
    occlusion: bool = False
    occ_half: int = 40

    # backbone weights
    # If True, will try to use torchvision pretrained weights (needs offline cache or HTTPS ok)
    use_pretrained: bool = True
