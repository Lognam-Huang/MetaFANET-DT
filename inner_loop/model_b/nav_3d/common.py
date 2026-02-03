from __future__ import annotations

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, TransformedEnv

from .nav3d_env import Nav3DEnv

# 可选：如果你还想保留 old 的 DebugNoneTransform
try:
    from .debug_none_transform import DebugNoneTransform
except Exception:
    DebugNoneTransform = None


class Nav3DTask(Task):
    """
    External Nav3D tasks living in metaRL_merged.

    Usage:
      task=nav3d/raleigh_city
    """
    RALEIGH_CITY = None

    @staticmethod
    def associated_class():
        return Nav3DClass


class Nav3DClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        cfg = dict(self.config)
        cfg["seed"] = seed
        cfg["device"] = device
        cfg["num_envs"] = num_envs

        def make_env():
            base_env = Nav3DEnv(**cfg)
            # 调试：检查是否出现 None / 结构异常
            if DebugNoneTransform is not None:
                env = TransformedEnv(base_env)
                env.append_transform(DebugNoneTransform(name="Nav3D_NoneCheck"))
                return env
            return base_env

        return make_env

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", 200)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        if hasattr(env, "agent_names"):
            return {"agents": list(env.agent_names)}
        raise RuntimeError("Nav3DEnv must define either `group_map` or `agent_names`.")

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        info_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "observation" in info_spec[group]:
                del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        return None

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        return "nav3d"

    def log_info(self, batch: TensorDictBase):
        return {}
