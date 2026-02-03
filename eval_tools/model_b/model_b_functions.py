from __future__ import annotations

import os
import json
import numpy as np
import torch

from benchmarl.hydra_config import reload_experiment_from_file

from dataclasses import dataclass
from typing import Any, Callable

# -----------------------------
# Helpers
# -----------------------------
def _is_tensordict(x) -> bool:
    return x.__class__.__name__ in ("TensorDict", "TensorDictBase")


def _td_get(td, key, default=None):
    """
    Safe get for dict/TensorDict-like.
    key can be:
      - string: "reward"
      - tuple path: ("next", "reward") for TensorDict
    """
    try:
        if _is_tensordict(td):
            if isinstance(key, tuple):
                return td.get(key, default)
            return td.get(key, default)
        if isinstance(td, dict):
            if isinstance(key, tuple):
                cur = td
                for k in key:
                    if not isinstance(cur, dict) or k not in cur:
                        return default
                    cur = cur[k]
                return cur
            return td.get(key, default)
    except Exception:
        return default
    return default


def _to_numpy(x):
    """Convert torch tensor / numpy / list into numpy array (best-effort)."""
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return x


def _iter_sub_envs(env) -> list:
    """
    Best-effort to extract sub-envs from vectorized/parallel env wrappers.
    Returns list of envs; if cannot find, returns [env].
    """
    # common names: envs, _envs, _env, base_env, unwrapped, etc.
    for attr in ("envs", "_envs"):
        if hasattr(env, attr):
            sub = getattr(env, attr)
            if isinstance(sub, (list, tuple)) and len(sub) > 0:
                return list(sub)

    # TorchRL ParallelEnv sometimes uses "envs" too, or stores in "_env"
    if hasattr(env, "_env") and getattr(env, "_env") is not None:
        inner = getattr(env, "_env")
        # inner might itself be a list
        if isinstance(inner, (list, tuple)) and len(inner) > 0:
            return list(inner)

    # Gymnasium vector env: env.envs
    if hasattr(env, "env") and getattr(env, "env") is not None:
        inner = getattr(env, "env")
        if hasattr(inner, "envs") and isinstance(inner.envs, (list, tuple)) and len(inner.envs) > 0:
            return list(inner.envs)

    return [env]


def _try_call(obj, fn_names: list[str], *args, **kwargs) -> bool:
    """
    Try calling obj.<fn>(*args, **kwargs) for fn in fn_names.
    Return True if succeeded.
    """
    for name in fn_names:
        if hasattr(obj, name):
            fn = getattr(obj, name)
            if callable(fn):
                try:
                    fn(*args, **kwargs)
                    return True
                except Exception:
                    continue
    return False


def _best_effort_extract_positions(obs_or_td) -> dict:
    """
    Try to extract uav positions / target positions from observation.
    Returns dict with keys: uav_pos, targets (may be None).
    """
    out = {"uav_pos": None, "targets": None}

    # Common patterns in dict obs
    if isinstance(obs_or_td, dict):
        # direct keys
        for k in ("uav_pos", "uavs_pos", "uav_positions", "uavs_positions"):
            if k in obs_or_td:
                out["uav_pos"] = _to_numpy(obs_or_td[k])
                break
        for k in ("targets", "target_pos", "target_positions", "targets_pos"):
            if k in obs_or_td:
                out["targets"] = _to_numpy(obs_or_td[k])
                break

    # TensorDict patterns: try a few common keys
    if _is_tensordict(obs_or_td):
        # sometimes obs stored under ("agents","state","pos") etc - can't know, try common names
        cand_uav_keys = [
            "uav_pos", "uavs_pos", "uav_positions", "uavs_positions",
            ("observation", "uav_pos"), ("observation", "uavs_pos"),
            ("agents", "pos"), ("agents", "position"),
        ]
        for k in cand_uav_keys:
            v = _td_get(obs_or_td, k, None)
            if v is not None:
                out["uav_pos"] = _to_numpy(v)
                break

        cand_t_keys = [
            "targets", "target_pos", "target_positions", "targets_pos",
            ("observation", "targets"), ("observation", "target_pos"),
            ("targets", "pos"), ("targets", "position"),
        ]
        for k in cand_t_keys:
            v = _td_get(obs_or_td, k, None)
            if v is not None:
                out["targets"] = _to_numpy(v)
                break

    return out

def get_positions_from_env(env, td=None):
    import numpy as np
    import torch

    def _as_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    # unwrap candidates
    candidates = [env]
    u = getattr(env, "unwrapped", None)
    if u is not None:
        candidates.append(u)

    # 1) explicit getters (preferred)
    for obj in candidates:
        if hasattr(obj, "get_uav_positions") and hasattr(obj, "get_target_positions"):
            try:
                uav = _as_np(obj.get_uav_positions())
                tar = _as_np(obj.get_target_positions())
                if uav is not None and tar is not None:
                    return uav, tar
            except Exception:
                pass

    # 2) optional: parse from td (only if you want)
    # ... 这里可以留空或保留你之前的 td 解析逻辑

    return None, None


# -----------------------------
# Core: apply case to env
# -----------------------------
def apply_eval_case_to_env(env, case: dict, *, apply_to_all_envs: bool = False, log: Callable = print):
    """
    Apply a single eval case (fixed init positions) to env.

    case schema:
      case["uav_pos"] : list[list[3]]  (num_uavs x 3)
      case["targets"] : list[list[3]]  (num_targets x 3)
      optional: case["env_idx"] : int

    This function tries multiple common APIs. If your env doesn't provide any, it raises
    an actionable error telling you what minimal method to add.
    """
    if case is None:
        return

    uav_pos = case.get("uav_pos", None)
    targets = case.get("targets", None)
    env_idx = int(case.get("env_idx", 0))

    if uav_pos is None or targets is None:
        raise ValueError("case must contain 'uav_pos' and 'targets'")

    uav_pos = _to_numpy(uav_pos)
    targets = _to_numpy(targets)

    # Decide which env instances to apply
    env_list = _iter_sub_envs(env) if apply_to_all_envs else [env_list := None]  # dummy to satisfy linter
    if apply_to_all_envs:
        env_list = _iter_sub_envs(env)
    else:
        env_list = [env]

    # try apply on each env
    for i, e in enumerate(env_list):
        # prefer unwrapped if exists
        u = getattr(e, "unwrapped", None)
        target_objs = [e] + ([u] if u is not None else [])

        applied = False

        # 0) Best: a single dedicated method
        for obj in target_objs:
            if obj is None:
                continue
            if _try_call(obj, ["set_eval_case", "apply_eval_case", "set_case"], uav_pos, targets, env_idx=env_idx):
                applied = True
                break
            # Some envs accept dict
            if _try_call(obj, ["set_eval_case", "apply_eval_case", "set_case"], {"uav_pos": uav_pos, "targets": targets, "env_idx": env_idx}):
                applied = True
                break
        if applied:
            continue

        # 1) Try separate setters
        for obj in target_objs:
            if obj is None:
                continue
            ok_uav = _try_call(obj, ["set_uav_positions", "set_uavs_positions", "set_uav_pos", "set_uavs_pos"], uav_pos, env_idx=env_idx)
            if not ok_uav:
                ok_uav = _try_call(obj, ["set_uav_positions", "set_uavs_positions", "set_uav_pos", "set_uavs_pos"], uav_pos)
            ok_t = _try_call(obj, ["set_targets", "set_target_positions", "set_targets_positions", "set_target_pos"], targets, env_idx=env_idx)
            if not ok_t:
                ok_t = _try_call(obj, ["set_targets", "set_target_positions", "set_targets_positions", "set_target_pos"], targets)

            if ok_uav and ok_t:
                applied = True
                break
        if applied:
            continue

        # 2) Try setting attributes directly (last resort)
        for obj in target_objs:
            if obj is None:
                continue
            # common attribute names
            attr_uav = ["uav_pos", "uavs_pos", "uav_positions", "uavs_positions", "_uav_pos", "_uavs_pos"]
            attr_t = ["targets", "target_pos", "target_positions", "targets_pos", "_targets", "_target_pos"]
            set_u = False
            set_t = False
            for a in attr_uav:
                if hasattr(obj, a):
                    try:
                        setattr(obj, a, uav_pos)
                        set_u = True
                        break
                    except Exception:
                        pass
            for a in attr_t:
                if hasattr(obj, a):
                    try:
                        setattr(obj, a, targets)
                        set_t = True
                        break
                    except Exception:
                        pass
            if set_u and set_t:
                applied = True
                break
        if applied:
            continue

        # If we reach here, we cannot apply
        raise RuntimeError(
            "无法把 eval case 注入到 env（找不到任何可用的 setter）。\n"
            f"- env type: {type(e)}\n"
            f"- apply_to_all_envs={apply_to_all_envs} (this env idx in list: {i})\n"
            "你需要在 Nav3DEnv（或其 unwrapped）里增加一个最小接口，例如：\n"
            "  def set_eval_case(self, uav_pos, targets, env_idx=0):\n"
            "      # 写入内部 state，并确保下一次观测/step 使用该 state\n"
            "或者分别实现：set_uav_positions(...) 和 set_targets(...)\n"
        )

    log(f"[apply_case] applied case id={case.get('id','(no id)')} env_idx={env_idx} uavs={len(uav_pos)} targets={len(targets)}")

# -----------------------------
# Core: rollout and collect
# -----------------------------
def rollout_and_collect(
    experiment,
    episodes: int = 20,
    max_steps: int = 200,
    *,
    case: dict | None = None,
    cases: list[dict] | None = None,
    buildings: Any | None = None,   # reserved for future use (collision/plot overlay)
    scenario: dict | None = None,   # legacy: {"uav_pos":..., "targets":..., "env_idx":...}
    apply_to_all_envs: bool = False,
    deterministic: bool = True,
    log: Callable = print,
):
    """
    Rollout the experiment policy in experiment.test_env and collect trajectories.

    Modes:
      - case provided: apply fixed init state for each episode
      - cases provided: cycle through cases
      - scenario provided (legacy): treated as a single case
      - none: pure random reset

    Returns:
      results: dict suitable for npz saving
        - episode_return: (episodes,)
        - episode_length: (episodes,)
        - success: (episodes,)  (best-effort)
        - traj: object array (len=episodes) of dict:
            rewards, dones, uav_pos (if detectable), targets (if detectable)
    """
    if not hasattr(experiment, "test_env") or experiment.test_env is None:
        raise ValueError("experiment.test_env is missing. Reload experiment with evaluation env enabled.")

    env = experiment.test_env

    # BenchMARL experiment typically exposes .policy
    policy = getattr(experiment, "policy", None)
    if policy is None:
        raise ValueError("experiment.policy not found. Cannot act for rollout.")

    # Unify legacy scenario into case
    if case is None and scenario is not None:
        case = dict(scenario)
        case.setdefault("id", "legacy_scenario")

    if case is not None and cases is not None:
        raise ValueError("Provide either case or cases, not both.")
    if cases is not None and len(cases) == 0:
        cases = None

    ep_returns = np.zeros((episodes,), dtype=np.float32)
    ep_lengths = np.zeros((episodes,), dtype=np.int32)
    ep_success = np.zeros((episodes,), dtype=np.int8)
    traj_list: list[dict] = []

    for ep in range(episodes):
        # Choose case for this episode
        this_case = None
        if case is not None:
            this_case = case
        elif cases is not None:
            this_case = cases[ep % len(cases)]

        

        # Apply fixed init after reset
        if this_case is not None:
            apply_eval_case_to_env(env, this_case, apply_to_all_envs=apply_to_all_envs, log=log)
            # optional refresh hooks
            _try_call(getattr(env, "unwrapped", env), ["refresh_obs", "update_obs", "_update_obs"])

        # Reset
        obs = env.reset()
        
        rewards = []
        dones = []
        uav_pos_steps = []
        targets_steps = []

        total_r = 0.0
        done = False
        last_step_out = None  # for success/info extraction

        for t in range(max_steps):
            # =========================================================
            # TorchRL path: EnvBase expects TensorDict input to step()
            # =========================================================
            if _is_tensordict(obs):
                td = obs

                # policy(td) should return a tensordict containing the action
                # try:
                #     td_act = policy(td, deterministic=deterministic)
                # except TypeError:
                #     td_act = policy(td)

                # Make policy deterministic-ish: use eval mode + no_grad
                # (True deterministic sampling depends on the distribution module, but this avoids illegal kwargs.)
                try:
                    policy.eval()
                except Exception:
                    pass
                
                import torch
                with torch.no_grad():
                    td_act = policy(td)  # <-- NO kwargs!

                # IMPORTANT: step expects tensordict, not action tensor
                try:
                    step_out = env.step(td_act)
                except Exception as e:
                    raise RuntimeError(f"Env.step (TensorDict) failed at ep={ep}, t={t}: {repr(e)}")

                # AFTER step, read positions directly from env state
                uav_now, tar_now = get_positions_from_env(env)
                uav_pos_steps.append(uav_now)
                targets_steps.append(tar_now)

                
                last_step_out = step_out

                # next td
                next_td = _td_get(step_out, "next", None)
                if next_td is None:
                    # some envs might directly return next td
                    next_td = step_out

                # reward / done
                r = _td_get(step_out, ("next", "reward"), None)
                if r is None:
                    r = _td_get(step_out, "reward", None)
                r = _to_numpy(r)
                try:
                    r_scalar = float(np.sum(r))
                except Exception:
                    r_scalar = float(r) if r is not None else 0.0

                d = _td_get(step_out, ("next", "done"), None)
                if d is None:
                    d = _td_get(step_out, "done", None)
                d = _to_numpy(d)
                try:
                    done = bool(np.any(d))
                except Exception:
                    done = bool(d) if d is not None else False

                # record positions (best-effort)
                pos = _best_effort_extract_positions(next_td)
                uav_pos_steps.append(pos["uav_pos"])
                targets_steps.append(pos["targets"])

                rewards.append(r_scalar)
                dones.append(done)

                total_r += r_scalar
                obs = next_td

            # =========================================================
            # Fallback path: gym-like env.step(action)
            # =========================================================
            else:
                try:
                    out = policy(obs)
                    action = out.get("action") if isinstance(out, dict) else getattr(out, "action", None)
                except Exception as e:
                    raise RuntimeError(f"Policy action computation failed at ep={ep}, t={t}: {repr(e)}")

                try:
                    step_out = env.step(action)
                except Exception as e:
                    raise RuntimeError(f"Env.step failed at ep={ep}, t={t}: {repr(e)}")

                last_step_out = step_out

                # parse (obs, reward, done, info)
                if isinstance(step_out, (tuple, list)) and len(step_out) >= 3:
                    next_obs, r, d = step_out[0], step_out[1], step_out[2]
                else:
                    next_obs, r, d = step_out, 0.0, False

                r = _to_numpy(r)
                try:
                    r_scalar = float(np.sum(r))
                except Exception:
                    r_scalar = float(r) if r is not None else 0.0

                d = _to_numpy(d)
                try:
                    done = bool(np.any(d))
                except Exception:
                    done = bool(d) if d is not None else False

                pos = _best_effort_extract_positions(next_obs)
                uav_pos_steps.append(pos["uav_pos"])
                targets_steps.append(pos["targets"])

                rewards.append(r_scalar)
                dones.append(done)

                total_r += r_scalar
                obs = next_obs

            if done:
                break

        ep_returns[ep] = total_r
        ep_lengths[ep] = len(rewards)

        # success best-effort
        success = 0
        if last_step_out is not None:
            info = _td_get(last_step_out, ("next", "info"), None)
            if info is None:
                info = _td_get(last_step_out, "info", None)
            if isinstance(info, dict) and "success" in info:
                try:
                    success = int(bool(info["success"]))
                except Exception:
                    success = 0
        ep_success[ep] = success

        traj_list.append(
            {
                "case_id": (this_case.get("id") if this_case else None),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "dones": np.asarray(dones, dtype=np.bool_),
                "uav_pos": uav_pos_steps,
                "targets": targets_steps,
            }
        )

        log(f"[rollout] ep {ep+1}/{episodes} return={total_r:.3f} len={ep_lengths[ep]} case={traj_list[-1]['case_id']}")

    results = {
        "episode_return": ep_returns,
        "episode_length": ep_lengths,
        "success": ep_success,
        "traj": np.array(traj_list, dtype=object),
    }
    return results


def get_leaf_env(env):
    base = env
    while hasattr(base, "base_env"):
        base = base.base_env
    return base


def build_env_from_experiment(experiment, log=print):
    if hasattr(experiment, "test_env") and experiment.test_env is not None:
        log("[env] Using experiment.test_env")
        env = experiment.test_env
    elif hasattr(experiment, "env_func") and experiment.env_func is not None:
        log("[env] Using experiment.env_func() to build env")
        env = experiment.env_func()
    else:
        raise RuntimeError("Experiment has neither test_env nor env_func.")

    cfg = getattr(experiment, "config", None)
    if cfg is not None and hasattr(cfg, "experiment") and hasattr(cfg.experiment, "train_device"):
        device = cfg.experiment.train_device
    else:
        device = "cpu"

    env.to(device)
    return env, device


import json
from pathlib import Path

def load_buildings_json(path: str):
    """
    Load buildings-only JSON, e.g. raleigh_buildings.json.
    Returns: list[dict] or dict depending on file structure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"buildings json not found: {p}")
    with open(p, "r") as f:
        data = json.load(f)

    # Most of your city files are list of building dicts
    # We keep it flexible.
    return data


def load_eval_cases_yaml(path: str):
    """
    Load evaluation cases YAML defining UAV init positions and target positions.
    Expected schema:
      meta: {num_uavs, num_targets, ...}   # optional but recommended
      cases: - {id, uav_pos: [[x,y,z]...], targets: [[x,y,z]...], env_idx?}
    Returns dict: {"meta":..., "cases":[...]}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"eval cases yaml not found: {p}")

    # Prefer PyYAML if installed, else fallback to OmegaConf (already in your stack via Hydra)
    try:
        import yaml  # type: ignore
        with open(p, "r") as f:
            obj = yaml.safe_load(f)
    except Exception:
        from omegaconf import OmegaConf
        obj = OmegaConf.to_container(OmegaConf.load(p), resolve=True)

    if not isinstance(obj, dict):
        raise ValueError(f"eval cases yaml must be a dict at top-level, got: {type(obj)}")

    meta = obj.get("meta", {}) or {}
    cases = obj.get("cases", None)
    if cases is None:
        raise ValueError("eval cases yaml must contain top-level key: cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("eval cases yaml 'cases' must be a non-empty list")

    # Basic validation (lightweight)
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            raise ValueError(f"case[{i}] must be dict, got {type(c)}")
        if "id" not in c:
            raise ValueError(f"case[{i}] missing 'id'")
        if "uav_pos" not in c:
            raise ValueError(f"case[{i}] missing 'uav_pos'")
        if "targets" not in c:
            raise ValueError(f"case[{i}] missing 'targets'")
        c.setdefault("env_idx", 0)

    return {"meta": meta, "cases": cases}


def select_eval_cases(cases, include_ids=None, exclude_ids=None, pattern: str | None = None, limit: int | None = None):
    """
    Filter cases by ids / glob-like pattern / limit.
    - include_ids: list of ids to keep (None means keep all)
    - exclude_ids: list of ids to drop
    - pattern: simple substring match for now (可以后续改成 fnmatch)
    """
    out = list(cases)

    if include_ids:
        include_set = set(include_ids)
        out = [c for c in out if c.get("id") in include_set]

    if exclude_ids:
        exclude_set = set(exclude_ids)
        out = [c for c in out if c.get("id") not in exclude_set]

    if pattern:
        out = [c for c in out if pattern in str(c.get("id", ""))]

    if limit is not None:
        out = out[: int(limit)]

    return out


# # 兼容旧接口：如果你还想保留这个名字（可选）
# def load_scenario_json(path: str):
#     """
#     Legacy: scenario json with uav_pos + targets.
#     Kept for backward compatibility ONLY.
#     """
#     with open(path, "r") as f:
#         sc = json.load(f)
#     if "uav_pos" not in sc or "targets" not in sc:
#         raise ValueError(
#             "This file is not a legacy scenario-state JSON (missing 'uav_pos'/'targets'). "
#             "If this is a buildings file, use load_buildings_json(). "
#             "If you want eval cases, use load_eval_cases_yaml()."
#         )
#     sc.setdefault("env_idx", 0)
#     return sc



def reload_experiment(checkpoint_path: str):
    return reload_experiment_from_file(checkpoint_path)


def build_env_and_policy(experiment, log=print):
    env, device = build_env_from_experiment(experiment, log=log)
    leaf_env = get_leaf_env(env)

    if not hasattr(experiment, "policy"):
        raise RuntimeError("experiment has no attribute 'policy'.")
    policy = experiment.policy
    policy.to(device)
    policy.eval()

    return env, leaf_env, policy, device


def save_results_npz(results: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **results)


def load_results_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def extract_episode_traj(results: dict, ep: int = 0, env_idx: int = 0, use_all: bool = True):
    """
    Returns traj[T,N,3] for a selected episode/env_idx.
    - If use_all=True, use results['traj_all'][ep] which is [T,B,N,3]
    - Else, use legacy results['traj'][ep] which is [T,N,3] (env0)
    """
    if use_all and "traj_all" in results:
        traj_all = results["traj_all"][ep]  # object -> ndarray [T,B,N,3]
        return traj_all[:, env_idx, :, :]
    else:
        traj = results["traj"][ep]          # object -> ndarray [T,N,3]
        return traj
