import json
import numpy as np
import torch
import tensordict as td

from typing import Optional, Tuple

from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BoundedTensorSpec,
)


# -----------------------------
# Utilities
# -----------------------------
def load_buildings(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


# =====================================================================
#   Nav3DEnv — TorchRL / BenchMARL Compatible (Improved reset + collisions)
# =====================================================================
class Nav3DEnv(EnvBase):
    def __init__(
        self,
        json_path,
        num_uavs=5,
        max_steps=200,
        uav_speed=3.0,
        height_min=50.0,
        height_max=150.0,

        # -----------------------------
        # Scene bounds (preferred for reset/clamp)
        # -----------------------------
        scene_x_min: Optional[float] = None,
        scene_x_max: Optional[float] = None,
        scene_y_min: Optional[float] = None,
        scene_y_max: Optional[float] = None,

        # -----------------------------
        # Target height range (if None -> use UAV height range)
        # -----------------------------
        target_height_min: Optional[float] = None,
        target_height_max: Optional[float] = None,

        # -----------------------------
        # Reset rejection sampling controls
        # -----------------------------
        reset_rejection_max_tries: int = 50,
        reset_rejection_oversample: int = 4,

        # reward basics
        reward_step=-0.01,
        reward_goal=10.0,
        reward_collision=-20.0,
        reward_uav_collision=-10.0,

        # success settings
        success_radius=15.0,
        stay_near_bonus=0.5,
        goal_bonus_once=True,  # True: first reach only; False: each step in radius

        # distance shaping
        use_progress_reward=True,
        w_progress=1.0,
        use_distance_penalty=True,
        w_distance=-0.005,

        # collisions
        use_building_collision=True,
        uav_collision_radius=5.0,
        uav_collision_count_penalty=False,  # True: penalize by count, else boolean

        # termination
        terminate_on_all_success=True,

        # randomization
        randomize_start_positions=True,
        randomize_targets=True,

        # torchrl
        seed=None,
        device="cpu",
        num_envs=1,
        **kwargs,
    ):
        super().__init__(device=device, batch_size=[num_envs])

        self.num_envs = int(num_envs)
        self.num_uavs = int(num_uavs)
        self.max_steps = int(max_steps)

        self.uav_speed = float(uav_speed)
        self.height_min = float(height_min)
        self.height_max = float(height_max)

        # scene bounds (preferred)
        self.scene_x_min = scene_x_min
        self.scene_x_max = scene_x_max
        self.scene_y_min = scene_y_min
        self.scene_y_max = scene_y_max

        # target height range
        self.target_height_min = float(target_height_min) if target_height_min is not None else self.height_min
        self.target_height_max = float(target_height_max) if target_height_max is not None else self.height_max

        # reset rejection
        self.reset_rejection_max_tries = int(reset_rejection_max_tries)
        self.reset_rejection_oversample = int(reset_rejection_oversample)

        # reward parameters
        self.reward_step = float(reward_step)
        self.reward_goal = float(reward_goal)
        self.reward_collision = float(reward_collision)
        self.reward_uav_collision = float(reward_uav_collision)

        self.success_radius = float(success_radius)
        self.stay_near_bonus = float(stay_near_bonus)
        self.goal_bonus_once = bool(goal_bonus_once)

        self.use_progress_reward = bool(use_progress_reward)
        self.w_progress = float(w_progress)
        self.use_distance_penalty = bool(use_distance_penalty)
        self.w_distance = float(w_distance)

        self.use_building_collision = bool(use_building_collision)
        self.uav_collision_radius = float(uav_collision_radius)
        self.uav_collision_count_penalty = bool(uav_collision_count_penalty)

        self.terminate_on_all_success = bool(terminate_on_all_success)

        self.randomize_start_positions = bool(randomize_start_positions)
        self.randomize_targets = bool(randomize_targets)

        # buildings -> vectorized bounds tensor on device
        self.buildings = load_buildings(json_path)
        if len(self.buildings) > 0:
            bounds = []
            xs, ys = [], []
            for b in self.buildings:
                bounds.append([b["xmin"], b["ymin"], b["zmin"], b["xmax"], b["ymax"], b["zmax"]])
                xs += [b["xmin"], b["xmax"]]
                ys += [b["ymin"], b["ymax"]]
            self.building_bounds = torch.tensor(bounds, dtype=torch.float32, device=self.device)  # [M,6]
            # fallback boundary clamp derived from buildings (+ margin)
            self.boundary_margin = 200.0
            self.world_x_min = min(xs) - self.boundary_margin
            self.world_x_max = max(xs) + self.boundary_margin
            self.world_y_min = min(ys) - self.boundary_margin
            self.world_y_max = max(ys) + self.boundary_margin
        else:
            self.building_bounds = None
            self.boundary_margin = 200.0
            self.world_x_min = None
            self.world_x_max = None
            self.world_y_min = None
            self.world_y_max = None

        # multi-agent names
        self.agent_names = [f"uav_{i}" for i in range(self.num_uavs)]

        # -----------------------------
        # Specs (unbatched)
        # -----------------------------
        rel_other_dim = 6 * (self.num_uavs - 1)
        self.single_obs_dim = 3 + 3 + 3 + rel_other_dim

        full_obs_tensor_spec = UnboundedContinuousTensorSpec(
            shape=(self.num_uavs, self.single_obs_dim),
            device=self.device,
        )
        self.full_observation_spec_unbatched = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"observation": full_obs_tensor_spec},
                    shape=(self.num_uavs,),
                )
            }
        )

        full_act_tensor_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self.num_uavs, 3),
            device=self.device,
        )
        self.full_action_spec_unbatched = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": full_act_tensor_spec},
                    shape=(self.num_uavs,),
                )
            }
        )

        # -----------------------------
        # Internal state
        # -----------------------------
        self.uav_pos = None        # [B,N,3]
        self.uav_vel = None        # [B,N,3]
        self.targets = None        # [B,N,3]
        self.steps = None          # [B]
        self.prev_dist = None      # [B,N]
        self.reached_target = None # [B,N] bool

        # scenario injection
        self._pending_reset_scenario = None

        if seed is not None:
            self._set_seed(seed)

    # =================================================================
    # TorchRL required
    # =================================================================
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # =================================================================
    # Bounds helpers
    # =================================================================
    def _get_scene_bounds_xy(self) -> Tuple[float, float, float, float]:
        """
        Prefer explicit scene bounds; otherwise fallback to building-derived bounds;
        otherwise use a conservative default.
        """
        if (
            self.scene_x_min is not None and self.scene_x_max is not None and
            self.scene_y_min is not None and self.scene_y_max is not None
        ):
            return float(self.scene_x_min), float(self.scene_x_max), float(self.scene_y_min), float(self.scene_y_max)

        if self.world_x_min is not None:
            return float(self.world_x_min), float(self.world_x_max), float(self.world_y_min), float(self.world_y_max)

        return -1000.0, 1000.0, -1000.0, 1000.0

    def _sample_uniform_positions(self, B: int, K: int, z_min: float, z_max: float) -> torch.Tensor:
        """
        Sample [B,K,3] uniformly within scene bounds and z range.
        """
        x_min, x_max, y_min, y_max = self._get_scene_bounds_xy()
        x = torch.rand(B, K, device=self.device) * (x_max - x_min) + x_min
        y = torch.rand(B, K, device=self.device) * (y_max - y_min) + y_min
        z = torch.rand(B, K, device=self.device) * (z_max - z_min) + z_min
        return torch.stack([x, y, z], dim=-1)  # [B,K,3]

    def _inside_any_building(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Check whether each point is inside ANY building AABB.
        Args:
            pos: [B,K,3] float tensor on self.device
        Returns:
            inside: [B,K] bool tensor
        """
        if self.building_bounds is None or self.building_bounds.numel() == 0:
            return torch.zeros(pos.shape[0], pos.shape[1], dtype=torch.bool, device=self.device)

        bb = self.building_bounds.view(1, 1, -1, 6)  # [1,1,M,6]
        p = pos.unsqueeze(2)  # [B,K,1,3]

        in_x = (p[..., 0] >= bb[..., 0]) & (p[..., 0] <= bb[..., 3])
        in_y = (p[..., 1] >= bb[..., 1]) & (p[..., 1] <= bb[..., 4])
        in_z = (p[..., 2] >= bb[..., 2]) & (p[..., 2] <= bb[..., 5])

        inside = (in_x & in_y & in_z).any(dim=-1)  # [B,K]
        return inside

    def _sample_positions_outside_buildings(self, B: int, K: int, z_min: float, z_max: float) -> torch.Tensor:
        """
        Rejection sampling to ensure sampled positions are NOT inside any building.
        Returns [B,K,3].
        """
        if self.building_bounds is None or self.building_bounds.numel() == 0:
            return self._sample_uniform_positions(B, K, z_min, z_max)

        remaining = torch.ones(B, K, dtype=torch.bool, device=self.device)
        out = torch.empty(B, K, 3, dtype=torch.float32, device=self.device)

        tries = 0
        while remaining.any() and tries < self.reset_rejection_max_tries:
            tries += 1

            cand = self._sample_uniform_positions(
                B, K * self.reset_rejection_oversample, z_min, z_max
            )  # [B, K*R, 3]
            inside = self._inside_any_building(cand)  # [B, K*R]
            valid = ~inside  # [B, K*R]

            # Small loop over B (typically small), reset is not hot-path.
            for b in range(B):
                if not remaining[b].any():
                    continue
                valid_idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue

                slots = torch.nonzero(remaining[b], as_tuple=False).squeeze(-1)
                take = min(slots.numel(), valid_idx.numel())
                out[b, slots[:take]] = cand[b, valid_idx[:take]]
                remaining[b, slots[:take]] = False

        # If still remaining, fall back to uniform (or raise if you prefer strictness).
        if remaining.any():
            fallback = self._sample_uniform_positions(B, K, z_min, z_max)
            out[remaining] = fallback[remaining]

        return out

    # =================================================================
    # RESET
    # =================================================================
    def _reset(self, tensordict=None):
        B, N = self.num_envs, self.num_uavs

        self.steps = torch.zeros(B, dtype=torch.int, device=self.device)

        # init positions
        if self.randomize_start_positions:
            self.uav_pos = self._sample_positions_outside_buildings(B, N, self.height_min, self.height_max)
        else:
            self.uav_pos = torch.zeros(B, N, 3, device=self.device)
            self.uav_pos[..., 2] = (self.height_min + self.height_max) * 0.5

        self.uav_vel = torch.zeros(B, N, 3, device=self.device)

        # init targets
        if self.randomize_targets:
            self.targets = self._sample_positions_outside_buildings(B, N, self.target_height_min, self.target_height_max)
        else:
            self.targets = torch.zeros(B, N, 3, device=self.device)
            self.targets[..., 2] = self.target_height_min

        # apply scenario if requested
        if self._pending_reset_scenario is not None:
            sc = self._pending_reset_scenario
            self._pending_reset_scenario = None
            self._apply_scenario_now(
                uav_pos=sc["uav_pos"],
                targets=sc["targets"],
                env_idx=int(sc.get("env_idx", 0)),
                apply_to_all_envs=bool(sc.get("apply_to_all_envs", False)),
            )

        # clamp within boundaries
        self._clamp_positions()

        # init cached distances
        with torch.no_grad():
            dist = torch.linalg.norm(self.uav_pos - self.targets, dim=-1)  # [B,N]
            self.prev_dist = dist.detach().clone()
            self.reached_target = torch.zeros_like(dist, dtype=torch.bool)

        obs = self._make_obs()

        agent_reward = torch.zeros(B, N, device=self.device)
        agent_done = torch.zeros(B, N, dtype=torch.bool, device=self.device)

        global_reward = agent_reward.mean(dim=1, keepdim=True)  # [B,1]
        global_done = torch.zeros(B, 1, dtype=torch.bool, device=self.device)

        agents_td = td.TensorDict(
            {
                "observation": obs,  # [B,N,obs_dim]
                "action": torch.zeros(B, N, 3, device=self.device),
                "reward": agent_reward,
                "done": agent_done,
                "terminated": agent_done,
            },
            batch_size=[B, N],
            device=self.device,
        )

        td_out = td.TensorDict(
            {
                "agents": agents_td,
                "reward": global_reward,
                "done": global_done,
                "terminated": global_done,
            },
            batch_size=[B],
            device=self.device,
        )
        return td_out

    # =================================================================
    # STEP
    # =================================================================
    def _step(self, tensordict):
        B, N = self.num_envs, self.num_uavs

        actions = tensordict["agents", "action"]  # [B,N,3]

        # freeze reached targets (optional but stabilizing)
        if self.reached_target is not None:
            frozen_mask = self.reached_target.unsqueeze(-1)  # [B,N,1]
            effective_actions = torch.where(frozen_mask, torch.zeros_like(actions), actions)
        else:
            effective_actions = actions

        self.uav_vel = effective_actions * self.uav_speed
        self.uav_pos = self.uav_pos + self.uav_vel

        self._clamp_positions()

        # collisions
        building_col, uav_col, uav_col_count = self._check_collision()

        # reward
        agent_reward = self._compute_reward(building_col, uav_col, uav_col_count)  # [B,N]

        # termination: success or time limit
        self.steps += 1
        time_limit = (self.steps >= self.max_steps).unsqueeze(-1)  # [B,1]

        reached_hist = self.reached_target if self.reached_target is not None else torch.zeros(
            B, N, dtype=torch.bool, device=self.device
        )
        if self.terminate_on_all_success:
            success_env = reached_hist.all(dim=1, keepdim=True)
        else:
            success_env = reached_hist.any(dim=1, keepdim=True)

        global_done = time_limit | success_env  # [B,1]
        agent_done = global_done.expand(-1, N)  # [B,N]

        global_reward = agent_reward.mean(dim=1, keepdim=True)

        obs_next = self._make_obs()

        agents_td = td.TensorDict(
            {
                "observation": obs_next,
                "action": actions,  # store original actions
                "reward": agent_reward,
                "done": agent_done,
                "terminated": agent_done,
            },
            batch_size=[B, N],
            device=self.device,
        )

        td_out = td.TensorDict(
            {
                "agents": agents_td,
                "reward": global_reward,
                "done": global_done,
                "terminated": global_done,
            },
            batch_size=[B],
            device=self.device,
        )

        # sanity check
        for k, v in td_out.items(include_nested=True):
            if v is None:
                raise RuntimeError(f"[STEP] Found None in key: {k}")
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                raise RuntimeError(f"[STEP] Non-finite values in key: {k}")

        return td_out

    # =================================================================
    # Reward
    # =================================================================
    def _compute_reward(self, building_col, uav_col, uav_col_count):
        """
        Reward components (per agent):
          - step penalty: constant per step
          - progress reward: prev_dist - dist
          - distance penalty: w_distance * dist
          - collisions: building + uav-uav
          - success bonus: once (or per-step) within success_radius
          - stay-near bonus: small bonus while within radius
        """
        B, N = self.num_envs, self.num_uavs

        diff = self.uav_pos - self.targets
        dist = torch.linalg.norm(diff, dim=-1)  # [B,N]

        # 1) step penalty
        reward = torch.full((B, N), self.reward_step, device=self.device)

        # 2) progress reward
        if self.use_progress_reward:
            if self.prev_dist is None:
                self.prev_dist = dist.detach().clone()
            progress = self.prev_dist - dist
            reward = reward + self.w_progress * progress
            self.prev_dist = dist.detach()
        else:
            self.prev_dist = dist.detach()

        # 3) distance penalty
        if self.use_distance_penalty:
            reward = reward + self.w_distance * dist

        # 4) collisions
        if self.use_building_collision:
            reward = reward + building_col.float() * self.reward_collision

        if self.uav_collision_count_penalty:
            reward = reward + uav_col_count.float() * self.reward_uav_collision
        else:
            reward = reward + uav_col.float() * self.reward_uav_collision

        # 5) success detection
        reached_now = dist < self.success_radius  # [B,N] bool
        if self.reached_target is None:
            self.reached_target = torch.zeros_like(reached_now)

        if self.goal_bonus_once:
            newly_reached = reached_now & (~self.reached_target)
            reward = reward + newly_reached.float() * self.reward_goal
        else:
            reward = reward + reached_now.float() * self.reward_goal

        # update reached history
        self.reached_target = self.reached_target | reached_now

        # 6) stay-near bonus
        reward = reward + reached_now.float() * self.stay_near_bonus

        if not torch.isfinite(reward).all():
            raise RuntimeError("[Reward] Found NaN/Inf in reward.")

        return reward

    # =================================================================
    # Collisions (Building AABB + UAV-UAV proximity)
    # =================================================================
    def _check_collision(self):
        B, N = self.num_envs, self.num_uavs

        # building collision (reuse unified inside check)
        if (self.building_bounds is None) or (self.building_bounds.numel() == 0):
            building_collision = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        else:
            building_collision = self._inside_any_building(self.uav_pos)  # [B,N]

        # UAV-UAV collision (proximity)
        dist_mat = torch.cdist(self.uav_pos, self.uav_pos)  # [B,N,N]
        dist_mat.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
        close_mat = dist_mat < self.uav_collision_radius
        uav_uav_collision = close_mat.any(dim=-1)  # [B,N]
        uav_uav_count = close_mat.sum(dim=-1).to(torch.int32)  # [B,N]

        return building_collision, uav_uav_collision, uav_uav_count

    # =================================================================
    # Observation builder (same layout as before)
    # =================================================================
    def _make_obs(self):
        rel_target = self.targets - self.uav_pos  # [B,N,3]

        rel_list = []
        for i in range(self.num_uavs):
            others_pos = torch.cat([self.uav_pos[:, :i], self.uav_pos[:, i + 1 :]], dim=1)
            others_vel = torch.cat([self.uav_vel[:, :i], self.uav_vel[:, i + 1 :]], dim=1)

            rel_p = others_pos - self.uav_pos[:, i : i + 1]
            rel_v = others_vel - self.uav_vel[:, i : i + 1]

            rel = torch.cat(
                [rel_p.reshape(self.num_envs, -1), rel_v.reshape(self.num_envs, -1)],
                dim=1,
            )
            rel_list.append(rel)

        rel_others = torch.stack(rel_list, dim=1)  # [B,N,6*(N-1)]

        # scaling
        pos_scale = 200.0
        vel_scale = 20.0

        uav_pos_scaled = self.uav_pos / pos_scale
        uav_vel_scaled = self.uav_vel / vel_scale
        rel_target_scaled = rel_target / pos_scale
        rel_others_scaled = rel_others / pos_scale

        obs = torch.cat([uav_pos_scaled, uav_vel_scaled, rel_target_scaled, rel_others_scaled], dim=-1)
        return obs

    # =================================================================
    # Helpers
    # =================================================================
    def _clamp_positions(self):
        # Prefer scene bounds for x/y clamp; fallback to building-derived bounds; else no x/y clamp.
        x_min, x_max, y_min, y_max = self._get_scene_bounds_xy()

        # If the bounds were from default fallback, you may still want clamp;
        # this is generally safer than leaving it unbounded.
        self.uav_pos[..., 0].clamp_(x_min, x_max)
        self.uav_pos[..., 1].clamp_(y_min, y_max)
        self.targets[..., 0].clamp_(x_min, x_max)
        self.targets[..., 1].clamp_(y_min, y_max)

        # z clamped
        self.uav_pos[..., 2].clamp_(self.height_min, self.height_max)
        self.targets[..., 2].clamp_(self.target_height_min, self.target_height_max)

    def set_next_reset_scenario(self, uav_pos, targets, env_idx: int = 0, apply_to_all_envs: bool = False):
        self._pending_reset_scenario = {
            "uav_pos": uav_pos,
            "targets": targets,
            "env_idx": env_idx,
            "apply_to_all_envs": apply_to_all_envs,
        }

    def _apply_scenario_now(self, uav_pos, targets, env_idx=0, apply_to_all_envs=False):
        device = self.device
        dtype = torch.float32

        up = torch.as_tensor(uav_pos, device=device, dtype=dtype)
        tg = torch.as_tensor(targets, device=device, dtype=dtype)

        if up.dim() == 2:
            up = up.unsqueeze(0)
        if tg.dim() == 2:
            tg = tg.unsqueeze(0)

        if apply_to_all_envs:
            if up.shape[0] == 1:
                self.uav_pos[:] = up.expand(self.num_envs, self.num_uavs, 3)
            else:
                self.uav_pos[:] = up
            if tg.shape[0] == 1:
                self.targets[:] = tg.expand(self.num_envs, self.num_uavs, 3)
            else:
                self.targets[:] = tg
        else:
            self.uav_pos[env_idx] = up[0] if up.shape[0] == 1 else up[env_idx]
            self.targets[env_idx] = tg[0] if tg.shape[0] == 1 else tg[env_idx]

        self._clamp_positions()

        with torch.no_grad():
            dist = torch.linalg.norm(self.uav_pos - self.targets, dim=-1)
            self.prev_dist = dist.detach().clone()
            self.reached_target = torch.zeros_like(dist, dtype=torch.bool)
