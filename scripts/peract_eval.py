import argparse
import os
import shutil
import ssl
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen


PERACT_REPO_URL = "https://github.com/peract/peract.git"
PERACT_CKPT_URL = "https://github.com/peract/peract/releases/download/v1.0.0/peract_600k.zip"

TASK_ALIASES = {
    "pickupcup": "pick_up_cup",
    "pick_up_cup": "pick_up_cup",
    "stackblocks": "stack_blocks",
    "stack_blocks": "stack_blocks",
    "open_drawer": "open_drawer",
}


def _read_tasks_from_config(path: Path):
    tasks = []
    in_rlbench = False
    in_tasks = False
    base_indent = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("rlbench:"):
            in_rlbench = True
            in_tasks = False
            base_indent = None
            continue
        if in_rlbench and line.lstrip().startswith("tasks:"):
            in_tasks = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if in_tasks:
            stripped = line.strip()
            if stripped.startswith("- "):
                tasks.append(stripped[2:].strip())
                continue
            if stripped == "":
                continue
            if len(line) - len(line.lstrip()) <= base_indent:
                break
    return tasks


def _normalize_tasks(raw_tasks):
    tasks = []
    for t in raw_tasks:
        key = t.strip().lower()
        if not key:
            continue
        tasks.append(TASK_ALIASES.get(key, key))
    return tasks


def _ensure_peract_repo(peract_root: Path, clone: bool):
    eval_py = peract_root / "eval.py"
    if eval_py.exists():
        return
    if not clone:
        raise RuntimeError(f"PerAct repo not found at {peract_root}. Use --clone to fetch it.")
    peract_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", PERACT_REPO_URL, str(peract_root)], check=True)


def _download_file(url: str, dest: Path, insecure: bool, ca_bundle: Optional[Path]):
    if ca_bundle is not None:
        context = ssl.create_default_context(cafile=str(ca_bundle))
    elif insecure:
        context = ssl._create_unverified_context()
    else:
        context = ssl.create_default_context()
    with urlopen(url, context=context) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _extract_zip(zip_path: Path, peract_root: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(peract_root)


def _ensure_checkpoint(peract_root: Path, ckpt_url: str, insecure: bool, ca_bundle: Optional[Path]):
    ckpt_cfg = peract_root / "ckpts" / "multi" / "PERACT_BC" / "seed0" / "config.yaml"
    if ckpt_cfg.exists():
        return ckpt_cfg
    peract_root.mkdir(parents=True, exist_ok=True)
    zip_path = peract_root / "peract_600k.zip"
    _download_file(ckpt_url, zip_path, insecure, ca_bundle)
    _extract_zip(zip_path, peract_root)
    zip_path.unlink(missing_ok=True)
    if not ckpt_cfg.exists():
        raise RuntimeError("Checkpoint unzip succeeded but config.yaml is missing.")
    return ckpt_cfg


def _patch_preprocess_agent(peract_root: Path, patch_src: Path):
    dst = peract_root / "helpers" / "preprocess_agent.py"
    if not dst.exists():
        raise RuntimeError(f"Missing preprocess_agent.py at {dst}")
    content = dst.read_text(encoding="utf-8")
    if "PERACT_OCCLUSION_PATCH" in content:
        return
    backup = dst.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(content, encoding="utf-8")
    dst.write_text(patch_src.read_text(encoding="utf-8"), encoding="utf-8")


def _build_eval_cmd(peract_root: Path,
                    tasks,
                    demo_path: Path,
                    eval_episodes: int,
                    gpu: int,
                    headless: bool):
    tasks_arg = "[" + ",".join(tasks) + "]"
    cmd = [
        sys.executable,
        "eval.py",
        f"rlbench.tasks={tasks_arg}",
        "rlbench.task_name=multi",
        f"rlbench.demo_path={demo_path.as_posix()}",
        f"framework.logdir={peract_root.as_posix()}/ckpts",
        f"framework.eval_episodes={eval_episodes}",
        "framework.eval_envs=1",
        "framework.eval_type=last",
        f"framework.gpu={gpu}",
        f"framework.env_gpu={gpu}",
        f"rlbench.headless={str(headless)}",
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run PerAct eval with RGB occlusion.")
    parser.add_argument("--peract-root", type=Path, default=Path(os.environ.get("PERACT_ROOT", "third_party/peract")))
    parser.add_argument("--demo-path", type=Path, default=None)
    parser.add_argument("--tasks", type=str, default="pickupcup,stackblocks,open_drawer")
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--windowed", action="store_true", default=False)
    parser.add_argument("--clone", action="store_true", help="Clone PerAct repo if missing.")
    parser.add_argument("--download-ckpt", action="store_true", help="Download PerAct checkpoint if missing.")
    parser.add_argument("--ckpt-url", type=str, default=PERACT_CKPT_URL)
    parser.add_argument("--ckpt-zip", type=Path, default=None, help="Use a local peract_600k.zip instead of downloading.")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL cert verification for checkpoint download.")
    parser.add_argument("--ca-bundle", type=Path, default=None, help="Custom CA bundle for SSL verification.")
    parser.add_argument("--no-occlude", action="store_true", help="Disable occlusion patch.")
    parser.add_argument("--occ-mode", type=str, default="center", choices=("center", "random"))
    parser.add_argument("--occ-half", type=int, default=32)
    parser.add_argument("--occ-prob", type=float, default=1.0)
    parser.add_argument("--occlude-point-cloud", action="store_true", default=False)
    parser.add_argument("--force-tasks", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    if args.windowed:
        args.headless = False

    peract_root = args.peract_root.resolve()
    _ensure_peract_repo(peract_root, args.clone)

    ckpt_cfg = None
    if args.ckpt_zip is not None:
        if not args.ckpt_zip.exists():
            raise RuntimeError(f"ckpt-zip not found: {args.ckpt_zip}")
        _extract_zip(args.ckpt_zip, peract_root)
        ckpt_cfg = peract_root / "ckpts" / "multi" / "PERACT_BC" / "seed0" / "config.yaml"
        if not ckpt_cfg.exists():
            raise RuntimeError("Checkpoint unzip succeeded but config.yaml is missing.")
    elif args.download_ckpt:
        ckpt_cfg = _ensure_checkpoint(peract_root, args.ckpt_url, args.insecure, args.ca_bundle)
    else:
        ckpt_cfg = peract_root / "ckpts" / "multi" / "PERACT_BC" / "seed0" / "config.yaml"
        if not ckpt_cfg.exists():
            raise RuntimeError("Checkpoint config missing. Use --download-ckpt to fetch it.")

    if not args.no_occlude:
        patch_src = Path(__file__).with_name("peract_preprocess_agent.py")
        _patch_preprocess_agent(peract_root, patch_src)

    raw_tasks = [t.strip() for t in args.tasks.split(",")]
    tasks = _normalize_tasks(raw_tasks)

    ckpt_tasks = _read_tasks_from_config(ckpt_cfg)
    missing = [t for t in tasks if t not in ckpt_tasks]
    if missing and not args.force_tasks:
        print("Warning: tasks not in checkpoint:", ", ".join(missing))
        tasks = [t for t in tasks if t in ckpt_tasks]
    if not tasks:
        raise RuntimeError("No valid tasks to evaluate after filtering.")

    demo_path = args.demo_path or (peract_root / "data" / "val")
    if not demo_path.exists():
        print(f"Warning: demo_path does not exist: {demo_path}")

    env = os.environ.copy()
    if not args.no_occlude:
        env["PERACT_OCCLUSION"] = "1"
        env["PERACT_OCCLUSION_MODE"] = args.occ_mode
        env["PERACT_OCCLUSION_HALF"] = str(args.occ_half)
        env["PERACT_OCCLUSION_PROB"] = str(args.occ_prob)
        if args.occlude_point_cloud:
            env["PERACT_OCCLUDE_POINT_CLOUD"] = "1"

    cmd = _build_eval_cmd(peract_root, tasks, demo_path, args.eval_episodes, args.gpu, args.headless)
    print("PerAct eval command:")
    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=peract_root, env=env, check=True)


if __name__ == "__main__":
    main()
