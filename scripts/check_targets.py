import numpy as np
from pathlib import Path


def main():
    root = Path("/rl/rlbench_bc/data/pickupcup")
    paths = sorted(root.glob("ep_*/targets.npy"))
    if not paths:
        print("No targets.npy found under", root)
        return

    ts = [np.load(p, mmap_mode="r") for p in paths]
    t = np.concatenate(ts, axis=0)

    print("shape:", t.shape)
    print("delta_xyz abs mean/max:", np.abs(t[:, :3]).mean(0), np.abs(t[:, :3]).max(0))
    print("delta_aa abs mean/max:", np.abs(t[:, 3:6]).mean(0), np.abs(t[:, 3:6]).max(0))
    print("gripper_open mean/min/max:", t[:, 6].mean(), t[:, 6].min(), t[:, 6].max())
    print("frac tiny delta_xyz (<1mm):", (np.linalg.norm(t[:, :3], axis=1) < 1e-3).mean())


if __name__ == "__main__":
    main()
