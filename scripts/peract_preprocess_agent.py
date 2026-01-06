from typing import List
import os
import torch

from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary, ImageSummary

# PERACT_OCCLUSION_PATCH


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if val == "":
        return default
    return val in ("1", "true", "yes", "y", "on")


class PreprocessAgent(Agent):

    def __init__(self,
                 pose_agent: Agent,
                 norm_rgb: bool = True):
        self._pose_agent = pose_agent
        self._norm_rgb = norm_rgb

        # Occlusion settings (eval-only by default)
        self._occ_enabled = _env_flag("PERACT_OCCLUSION", False)
        self._occ_mode = os.environ.get("PERACT_OCCLUSION_MODE", "center").strip().lower()
        self._occ_half = int(os.environ.get("PERACT_OCCLUSION_HALF", "32"))
        self._occ_prob = float(os.environ.get("PERACT_OCCLUSION_PROB", "1.0"))
        self._occ_value = float(os.environ.get("PERACT_OCCLUSION_VALUE", "0.0"))
        self._occ_apply_pcd = _env_flag("PERACT_OCCLUDE_POINT_CLOUD", False)

    def build(self, training: bool, device: torch.device = None):
        self._pose_agent.build(training, device)

    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def _should_occlude(self, device):
        if not self._occ_enabled:
            return False
        if self._occ_prob >= 1.0:
            return True
        return torch.rand((), device=device).item() < self._occ_prob

    def _occlude_single(self, img):
        # img: (C, H, W)
        _, h, w = img.shape
        half = max(1, min(self._occ_half, h // 2, w // 2))
        if not self._should_occlude(img.device):
            return img

        if self._occ_mode == "random":
            cx = int(torch.randint(half, w - half, (1,), device=img.device).item())
            cy = int(torch.randint(half, h - half, (1,), device=img.device).item())
        else:
            cx = w // 2
            cy = h // 2

        x0 = max(cx - half, 0)
        x1 = min(cx + half, w)
        y0 = max(cy - half, 0)
        y1 = min(cy + half, h)
        img[:, y0:y1, x0:x1] = self._occ_value
        return img

    def _occlude_tensor(self, tensor):
        if tensor.dim() == 3:
            return self._occlude_single(tensor)
        if tensor.dim() == 4:
            for b in range(tensor.shape[0]):
                self._occlude_single(tensor[b])
        return tensor

    def _maybe_occlude(self, key, tensor):
        if not self._occ_enabled:
            return tensor
        if 'rgb' in key:
            return self._occlude_tensor(tensor)
        if self._occ_apply_pcd and 'point_cloud' in key:
            return self._occlude_tensor(tensor)
        return tensor

    def update(self, step: int, replay_sample: dict) -> dict:
        # Samples are (B, N, ...) where N is number of buffers/tasks. This is a single task setup, so 0 index.
        replay_sample = {k: v[:, 0] if len(v.shape) > 2 else v for k, v in replay_sample.items()}
        for k, v in replay_sample.items():
            if self._norm_rgb and 'rgb' in k:
                v = self._norm_rgb_(v)
            else:
                v = v.float()
            v = self._maybe_occlude(k, v)
            replay_sample[k] = v
        self._replay_sample = replay_sample
        return self._pose_agent.update(step, replay_sample)

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        # observation = {k: torch.tensor(v) for k, v in observation.items()}
        for k, v in observation.items():
            if self._norm_rgb and 'rgb' in k:
                v = self._norm_rgb_(v)
            else:
                v = v.float()
            v = self._maybe_occlude(k, v)
            observation[k] = v
        act_res = self._pose_agent.act(step, observation, deterministic)
        act_res.replay_elements.update({'demo': False})
        return act_res

    def update_summaries(self) -> List[Summary]:
        prefix = 'inputs'
        demo_f = self._replay_sample['demo'].float()
        demo_proportion = demo_f.mean()
        tile = lambda x: torch.squeeze(
            torch.cat(x.split(1, dim=1), dim=-1), dim=1)
        sums = [
            ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
            HistogramSummary('%s/low_dim_state' % prefix,
                    self._replay_sample['low_dim_state']),
            HistogramSummary('%s/low_dim_state_tp1' % prefix,
                    self._replay_sample['low_dim_state_tp1']),
            ScalarSummary('%s/low_dim_state_mean' % prefix,
                    self._replay_sample['low_dim_state'].mean()),
            ScalarSummary('%s/low_dim_state_min' % prefix,
                    self._replay_sample['low_dim_state'].min()),
            ScalarSummary('%s/low_dim_state_max' % prefix,
                    self._replay_sample['low_dim_state'].max()),
            ScalarSummary('%s/timeouts' % prefix,
                    self._replay_sample['timeout'].float().mean()),
        ]

        for k, v in self._replay_sample.items():
            if 'rgb' in k or 'point_cloud' in k:
                if 'rgb' in k:
                    # Convert back to 0 - 1
                    v = (v + 1.0) / 2.0
                sums.append(ImageSummary('%s/%s' % (prefix, k), tile(v)))

        if 'sampling_probabilities' in self._replay_sample:
            sums.extend([
                HistogramSummary('replay/priority',
                                 self._replay_sample['sampling_probabilities']),
            ])
        sums.extend(self._pose_agent.update_summaries())
        return sums

    def act_summaries(self) -> List[Summary]:
        return self._pose_agent.act_summaries()

    def load_weights(self, savedir: str):
        self._pose_agent.load_weights(savedir)

    def save_weights(self, savedir: str):
        self._pose_agent.save_weights(savedir)

    def reset(self) -> None:
        self._pose_agent.reset()
