import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from config import Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlatDataset(Dataset):
    def __init__(self, root: Path):
        self.samples = []
        ep_dirs = sorted([p for p in root.glob("ep_*") if p.is_dir()])
        if not ep_dirs:
            raise RuntimeError(f"No episodes under {root}. Run collect_demos.py first.")

        for ep in ep_dirs:
            ip = ep / "images.npy"
            tp = ep / "targets.npy"
            if not ip.exists() or not tp.exists():
                continue
            imgs = np.load(ip, mmap_mode="r")
            tgts = np.load(tp, mmap_mode="r")
            Tlen = min(len(imgs), len(tgts))
            for t in range(Tlen):
                self.samples.append((str(ip), str(tp), t))

        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ip, tp, t = self.samples[idx]
        imgs = np.load(ip, mmap_mode="r")
        tgts = np.load(tp, mmap_mode="r")

        x = imgs[t]  # uint8 HWC
        y = tgts[t].astype(np.float32)  # (7,)

        x = self.tf(x)
        y = torch.from_numpy(y)
        return x, y


class DeltaPoseBC(nn.Module):
    def __init__(self, out_dim: int, use_pretrained: bool):
        super().__init__()
        if use_pretrained:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=None)

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # 512x1x1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


def main():
    cfg = Config()
    set_seed(cfg.seed)

    task_name = os.environ.get("TASK_NAME", "pickupcup")
    data_root = cfg.data_dir / task_name
    run_dir = cfg.work_dir / f"deltaposebc_{task_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] device:", device)

    ds = FlatDataset(data_root)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    model = DeltaPoseBC(out_dim=cfg.target_dim, use_pretrained=cfg.use_pretrained).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    # regression for delta_xyz (3) + delta_axis_angle (3), BCE for gripper_open
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(cfg.epochs):
        losses = []
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)  # (B,7)

            pred_delta = pred[:, :6]
            pred_glogit = pred[:, 6:7]

            gt_delta = y[:, :6]
            gt_g = y[:, 6:7].clamp(0.0, 1.0)

            loss = mse(pred_delta, gt_delta) + 0.2 * bce(pred_glogit, gt_g)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        ckpt = {"model": model.state_dict(), "task": task_name, "cfg": cfg.__dict__}
        torch.save(ckpt, run_dir / "ckpt_last.pt")

    print("[train] saved:", run_dir / "ckpt_last.pt")


if __name__ == "__main__":
    main()
