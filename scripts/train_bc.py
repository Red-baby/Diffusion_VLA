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
    def __init__(self, root: Path, use_image: bool, use_state: bool, normalize_state: bool, img_w: int, img_h: int):
        self.use_image = use_image
        self.use_state = use_state
        self.normalize_state = normalize_state
        self.img_w = img_w
        self.img_h = img_h

        self.samples = []
        self.state_paths = []
        self.state_dim = 0

        ep_dirs = sorted([p for p in root.glob("ep_*") if p.is_dir()])
        if not ep_dirs:
            raise RuntimeError(f"No episodes under {root}. Run collect_demos.py first.")

        for ep in ep_dirs:
            ip = ep / "images.npy"
            tp = ep / "targets.npy"
            sp = ep / "states.npy"

            if not tp.exists():
                continue
            if self.use_image and not ip.exists():
                continue
            if self.use_state and not sp.exists():
                raise RuntimeError(f"Missing states.npy in {ep}. Re-run collect_demos.py.")

            tgts = np.load(tp, mmap_mode="r")
            Tlen = len(tgts)

            if self.use_image:
                imgs = np.load(ip, mmap_mode="r")
                Tlen = min(Tlen, len(imgs))

            if self.use_state:
                states = np.load(sp, mmap_mode="r")
                Tlen = min(Tlen, len(states))
                self.state_paths.append(str(sp))

            for t in range(Tlen):
                self.samples.append((str(ip) if self.use_image else "", str(tp), str(sp) if self.use_state else "", t))

        if not self.samples:
            raise RuntimeError(f"No valid samples under {root}. Check data and config.")

        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        if self.use_state and self.state_paths:
            sample = np.load(self.state_paths[0], mmap_mode="r")
            if sample.ndim < 2:
                raise RuntimeError("states.npy should have shape (T, state_dim).")
            self.state_dim = int(sample.shape[1])

        self.state_mean = None
        self.state_std = None
        if self.use_state and self.normalize_state:
            self._compute_state_stats()

    def _compute_state_stats(self):
        uniq_paths = sorted(set(self.state_paths))
        states = [np.load(p) for p in uniq_paths]
        s = np.concatenate(states, axis=0).astype(np.float32)
        self.state_dim = int(s.shape[1])
        mean = s.mean(axis=0)
        std = s.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.state_mean = mean.astype(np.float32)
        self.state_std = std.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ip, tp, sp, t = self.samples[idx]
        tgts = np.load(tp, mmap_mode="r")

        if self.use_image:
            imgs = np.load(ip, mmap_mode="r")
            x = imgs[t].copy()  # uint8 HWC, ensure writable
            x = self.tf(x)
        else:
            x = torch.zeros((3, self.img_h, self.img_w), dtype=torch.float32)

        y = tgts[t].astype(np.float32)  # (7,)
        y = torch.from_numpy(y)

        if not self.use_state:
            return x, y

        states = np.load(sp, mmap_mode="r")
        s = states[t].astype(np.float32)
        if self.state_mean is not None:
            s = (s - self.state_mean) / self.state_std
        s = torch.from_numpy(s)
        return x, s, y


class DeltaPoseBC(nn.Module):
    def __init__(self, out_dim: int, state_dim: int, use_image: bool, use_state: bool, use_pretrained: bool):
        super().__init__()
        if not use_image and not use_state:
            raise ValueError("At least one of use_image/use_state must be True")

        self.use_image = use_image
        self.use_state = use_state

        img_dim = 0
        if self.use_image:
            if use_pretrained:
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                backbone = models.resnet18(weights=None)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # 512x1x1
            img_dim = 512

        state_dim_out = 0
        if self.use_state:
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            state_dim_out = 64

        head_in = img_dim + state_dim_out
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x=None, state=None):
        feats = []
        if self.use_image:
            if x is None:
                raise ValueError("Image input is required when use_image=True")
            img_feat = self.encoder(x)
            img_feat = torch.flatten(img_feat, 1)
            feats.append(img_feat)

        if self.use_state:
            if state is None:
                raise ValueError("State input is required when use_state=True")
            state_feat = self.state_mlp(state)
            feats.append(state_feat)

        if len(feats) == 1:
            feat = feats[0]
        else:
            feat = torch.cat(feats, dim=1)

        return self.head(feat)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    task_name = os.environ.get("TASK_NAME", "pickupcup")
    data_root = cfg.data_dir / task_name
    run_dir = cfg.work_dir / f"deltaposebc_{task_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] device:", device)

    ds = FlatDataset(
        data_root,
        use_image=cfg.use_image,
        use_state=cfg.use_state,
        normalize_state=cfg.normalize_state,
        img_w=cfg.img_w,
        img_h=cfg.img_h,
    )
    if cfg.use_state and ds.state_dim <= 0:
        raise RuntimeError("State dimension is invalid. Re-run collect_demos.py.")

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    model = DeltaPoseBC(
        out_dim=cfg.target_dim,
        state_dim=ds.state_dim,
        use_image=cfg.use_image,
        use_state=cfg.use_state,
        use_pretrained=cfg.use_pretrained,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    # regression for delta_xyz (3) + delta_axis_angle (3), BCE for gripper_open
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(cfg.epochs):
        losses = []
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            if cfg.use_state:
                x, s, y = batch
                s = s.to(device, non_blocking=True)
                if cfg.state_noise_std > 0.0:
                    s = s + torch.randn_like(s) * cfg.state_noise_std
            else:
                x, y = batch
                s = None

            if cfg.use_image:
                x = x.to(device, non_blocking=True)
            else:
                x = None

            y = y.to(device, non_blocking=True)

            pred = model(x, s)  # (B,7)

            pred_delta = torch.tanh(pred[:, :6])
            pred_glogit = pred[:, 6:7]

            gt_delta_xyz = y[:, :3]
            gt_delta_aa = y[:, 3:6]
            gt_g = y[:, 6:7].clamp(0.0, 1.0)

            gt_delta_xyz = torch.clamp(
                gt_delta_xyz / cfg.max_delta_xyz, -1.0, 1.0
            )
            aa_norm = torch.norm(gt_delta_aa, dim=1, keepdim=True)
            scale = torch.clamp(cfg.max_delta_angle / (aa_norm + 1e-8), max=1.0)
            gt_delta_aa = gt_delta_aa * scale
            gt_delta_aa = torch.clamp(
                gt_delta_aa / cfg.max_delta_angle, -1.0, 1.0
            )

            gt_delta = torch.cat([gt_delta_xyz, gt_delta_aa], dim=1)

            loss = mse(pred_delta, gt_delta) + 0.2 * bce(pred_glogit, gt_g)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        ckpt_cfg = cfg.__dict__.copy()
        ckpt_cfg["state_dim"] = ds.state_dim
        ckpt = {"model": model.state_dict(), "task": task_name, "cfg": ckpt_cfg}
        if ds.state_mean is not None:
            ckpt["state_mean"] = ds.state_mean
            ckpt["state_std"] = ds.state_std
        torch.save(ckpt, run_dir / "ckpt_last.pt")

    print("[train] saved:", run_dir / "ckpt_last.pt")


if __name__ == "__main__":
    main()
