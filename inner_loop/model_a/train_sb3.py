# train_sb3.py
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# ---- your custom env ----
from envs.placement_env import SionnaPlacementEnv


# -----------------------------
# helpers
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(p: str | Path) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)


def now_run_id() -> str:
    # Example: 20260113_161700
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def snapshot_config(cfg: Dict[str, Any], cfg_path: str, run_dir: Path) -> None:
    """Save a copy of config + a small run manifest into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) save resolved config json
    cfg_out = run_dir / "config_snapshot.json"
    with open(cfg_out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # 2) copy original config file (as-is)
    try:
        shutil.copy2(cfg_path, run_dir / Path(cfg_path).name)
    except Exception:
        pass

    # 3) save small manifest
    manifest = []
    manifest.append(f"timestamp: {datetime.now().isoformat()}")
    manifest.append(f"cfg_path: {cfg_path}")
    manifest.append(f"cwd: {os.getcwd()}")
    manifest.append(f"python: {sys.executable}")
    manifest.append(f"argv: {' '.join(sys.argv)}")
    save_text(run_dir / "run_manifest.txt", "\n".join(manifest) + "\n")


def get_algo_class(algo_name: str):
    """
    Return the SB3 algorithm class. Supports SB3 core + sb3-contrib algorithms.

    SB3 core: A2C, PPO, DQN, DDPG, TD3, SAC
    contrib: ARS, QRDQN, RecurrentPPO, TQC, TRPO, MaskablePPO, CrossQ (if present)
    """
    algo = algo_name.strip().upper()

    # ---- SB3 core ----
    try:
        if algo == "A2C":
            from stable_baselines3 import A2C
            return A2C
        if algo == "PPO":
            from stable_baselines3 import PPO
            return PPO
        if algo == "DQN":
            from stable_baselines3 import DQN
            return DQN
        if algo == "DDPG":
            from stable_baselines3 import DDPG
            return DDPG
        if algo == "TD3":
            from stable_baselines3 import TD3
            return TD3
        if algo == "SAC":
            from stable_baselines3 import SAC
            return SAC
    except Exception as e:
        raise RuntimeError(f"Failed importing SB3 core algo={algo}: {e}") from e

    # ---- sb3-contrib ----
    try:
        import sb3_contrib  # noqa: F401
        if algo in ["QRDQN", "QR-DQN"]:
            from sb3_contrib import QRDQN
            return QRDQN
        if algo == "ARS":
            from sb3_contrib import ARS
            return ARS
        if algo == "RECURRENTPPO":
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO
        if algo == "TQC":
            from sb3_contrib import TQC
            return TQC
        if algo == "TRPO":
            from sb3_contrib import TRPO
            return TRPO
        if algo in ["MASKABLEPPO", "MASKABLE_PPO"]:
            from sb3_contrib import MaskablePPO
            return MaskablePPO
        if algo == "CROSSQ":
            from sb3_contrib import CrossQ  # type: ignore
            return CrossQ
    except ImportError as e:
        raise RuntimeError(
            f"algo='{algo}' seems to require sb3-contrib, but it is not installed.\n"
            f"Install it via: pip install sb3-contrib\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed importing contrib algo={algo}: {e}") from e

    raise ValueError(
        f"Unknown algo='{algo}'. Supported examples: "
        f"A2C/PPO/DQN/DDPG/TD3/SAC and contrib ARS/QRDQN/RecurrentPPO/TQC/TRPO/MaskablePPO/CrossQ."
    )


def try_init_wandb(cfg: Dict[str, Any], run_dir: Path) -> Optional[Any]:
    """
    If cfg['wandb']['enabled'] is True, initialize wandb and return (run, WandbCallback).
    Otherwise return None.

    Notes:
      - sync_tensorboard=True will sync SB3 logger scalars to W&B.
      - you will still need an env-info callback to push custom info keys into TB/W&B.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except Exception as e:
        raise RuntimeError(f"wandb is enabled in cfg but cannot import wandb: {e}") from e

    project = wandb_cfg.get("project", "metaRL_modelA")
    run_name = wandb_cfg.get("name", None) or run_dir.name
    tags = wandb_cfg.get("tags", [])

    # Put wandb files inside this run_dir to keep things tidy
    wandb_dir = wandb_cfg.get("dir", None) or str(run_dir)

    run = wandb.init(
        project=project,
        name=run_name,
        tags=tags,
        config=cfg,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        dir=wandb_dir,
    )

    # Optional: let wandb callback also save model periodically (not required if you use CheckpointCallback)
    model_save_path = wandb_cfg.get("model_save_path", None)
    model_save_freq = int(wandb_cfg.get("model_save_freq", 0))
    if model_save_path is None:
        # default: a subfolder
        model_save_path = str(run_dir / "wandb_models")

    cb = WandbCallback(
        model_save_path=model_save_path,
        model_save_freq=model_save_freq,
        verbose=int(wandb_cfg.get("verbose", 1)),
    )
    return (run, cb)


# -----------------------------
# callbacks
# -----------------------------
class EnvInfoLoggerCallback(BaseCallback):
    """
    Log env.step() info dict keys into SB3 logger (TensorBoard/W&B via sync).

    It expects DummyVecEnv-style infos: list[dict] with length = n_envs.
    """

    def __init__(
        self,
        keys: List[str],
        prefix: str = "env",
        log_freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.keys = keys
        self.prefix = prefix
        self.log_freq = max(1, int(log_freq))
        self._n_calls = 0

    def _on_step(self) -> bool:
        self._n_calls += 1
        if (self._n_calls % self.log_freq) != 0:
            return True

        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        # infos is usually a list of dicts (vec env)
        if isinstance(infos, dict):
            infos_list = [infos]
        else:
            infos_list = list(infos)

        if len(infos_list) == 0:
            return True

        info0 = infos_list[0]
        for k in self.keys:
            if k in info0 and info0[k] is not None:
                v = info0[k]
                # Handle arrays (e.g., per_uav_load) by summarizing
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32).reshape(-1)
                    if arr.size > 0:
                        self.logger.record(f"{self.prefix}/{k}_mean", float(np.mean(arr)))
                        self.logger.record(f"{self.prefix}/{k}_min", float(np.min(arr)))
                        self.logger.record(f"{self.prefix}/{k}_max", float(np.max(arr)))
                else:
                    try:
                        self.logger.record(f"{self.prefix}/{k}", float(v))
                    except Exception:
                        # skip non-numeric
                        pass

        return True


# -----------------------------
# main
# -----------------------------
def main(cfg_path: str, seed: int = 0, do_check_env: bool = True):
    cfg_path = str(Path(cfg_path).resolve())
    cfg = load_json(cfg_path)

    # --- seed ---
    set_random_seed(seed)
    np.random.seed(seed)

    # --- base logging dirs from cfg ---
    log_cfg = cfg.get("log", {})
    tb_base = Path(ensure_dir(log_cfg.get("tb_dir", "runs/tb")))
    save_base = Path(ensure_dir(log_cfg.get("save_dir", "runs/modelA_sb3")))

    # --- run naming: algo + timestamp ---
    sb3_cfg = cfg.get("sb3", {})
    algo_name = str(sb3_cfg.get("algo", "SAC")).upper()
    run_id = now_run_id()
    run_name = f"{algo_name}_{run_id}"

    # --- run directories ---
    run_dir = save_base / run_name
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    tb_dir = tb_base / run_name
    ensure_dir(tb_dir)

    # --- snapshot config into run dir ---
    snapshot_config(cfg, cfg_path, run_dir)

    # --- create env ---
    env = SionnaPlacementEnv(cfg)
    env = Monitor(env)

    if do_check_env:
        check_env(env.unwrapped, warn=True)

    # --- algo selection (from cfg) ---
    policy_name = sb3_cfg.get("policy", "MlpPolicy")
    algo_kwargs = sb3_cfg.get("algo_kwargs", {})

    AlgoCls = get_algo_class(algo_name)

    # --- W&B (optional) ---
    wandb_pack = try_init_wandb(cfg, run_dir)
    wandb_run = None
    wandb_cb = None
    if wandb_pack is not None:
        wandb_run, wandb_cb = wandb_pack

    # --- train settings ---
    train_cfg = cfg.get("train", {})
    total_timesteps = int(train_cfg.get("total_timesteps", 100_000))
    save_freq = int(train_cfg.get("save_freq", 50_000))
    progress_bar = bool(train_cfg.get("progress_bar", True))

    # --- callbacks ---
    callbacks: List[BaseCallback] = []

    # 1) checkpoint callback
    if save_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(ckpt_dir),
                name_prefix=f"{algo_name.lower()}",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
        )

    # 2) env info logger callback (these will appear in TB/W&B)
    #    You can add more keys later if your env returns them.
    env_info_keys = ["coverage_count", "load_var", "invalid_uav_count", "per_uav_load"]
    callbacks.append(EnvInfoLoggerCallback(keys=env_info_keys, prefix="env", log_freq=1))

    # 3) wandb callback (optional)
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    cb = CallbackList(callbacks) if len(callbacks) > 0 else None

    # --- build model ---
    model = AlgoCls(
        policy_name,
        env,
        tensorboard_log=str(tb_dir),
        verbose=int(sb3_cfg.get("verbose", 1)),
        seed=seed,
        **algo_kwargs,
    )

    # --- train with safe interrupt handling ---
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=cb,
            progress_bar=progress_bar,
        )
    except KeyboardInterrupt:
        # Save an "interrupt" snapshot so you don't lose everything
        interrupt_path = run_dir / f"{algo_name.lower()}_interrupt.zip"
        model.save(str(interrupt_path))
        print(f"[INTERRUPTED] Saved interrupt model to: {interrupt_path}")
        raise
    finally:
        elapsed = time.time() - start_time
        save_text(run_dir / "train_elapsed_seconds.txt", f"{elapsed:.3f}\n")

    # --- final save ---
    final_path = run_dir / f"{algo_name.lower()}_final.zip"
    model.save(str(final_path))
    print(f"[DONE] Saved final model to: {final_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-check-env", action="store_true")
    args = parser.parse_args()

    main(args.cfg, seed=args.seed, do_check_env=(not args.no_check_env))
