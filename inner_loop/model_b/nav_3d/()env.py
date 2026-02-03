# import torch
# from tensordict import TensorDict
# # from benchmarl.environments.common import Env
# from benchmarl.environments.common import Environment
# from .common import Nav3DClass

# class Nav3DEnv(Env):
#     def __init__(self, task, **kwargs):
#         """
#         Wrapper that integrates Nav3DClass with BenchMARL.
#         将 Nav3DClass 与 BenchMARL 集成的包装器。
#         """
#         super().__init__(worker_index=kwargs.get("worker_index", 0), world_size=kwargs.get("world_size", 1))
#         self.task = task
#         self.config = task.config
        
#         # Initialize the physical logic from common.py
#         # 初始化 common.py 中的物理逻辑
#         self.engine = Nav3DClass(self.config)
        
#         # Internal state
#         self.uav_pos = None
#         self.targets = None

#     def _reset(self, tensordict: TensorDict) -> TensorDict:
#         """
#         Resets the environment for a new episode.
#         重置环境以开启新的回合。
#         """
#         # Logic to randomize positions within boundaries
#         # 在边界内随机初始化位置的逻辑
#         self.uav_pos = torch.rand((self.engine.num_uavs, 3)) # Placeholder
#         self.targets = torch.rand((self.engine.num_uavs, 3)) # Placeholder
        
#         return self._make_observation()

#     def _step(self, tensordict: TensorDict) -> TensorDict:
#         """
#         Executes one movement step.
#         执行一个移动步。
#         """
#         actions = tensordict["agents", "action"] # [N, 3]
        
#         # Update positions: pos = pos + action * speed
#         # 更新位置：位置 = 原位置 + 动作 * 速度
#         self.uav_pos += actions * self.engine.uav_speed
        
#         # Check collisions and compute rewards using the engine
#         # 使用引擎检查碰撞并计算奖励
#         b_col = self.engine.check_building_collisions(self.uav_pos)
#         reward = self.engine.compute_reward(self.uav_pos, self.targets, b_col, torch.zeros_like(b_col), torch.zeros_like(b_col))
        
#         out = self._make_observation()
#         out.set(("next", "reward"), reward.unsqueeze(-1))
#         out.set(("next", "done"), torch.zeros((self.engine.num_uavs, 1), dtype=torch.bool))
#         return out

#     def _make_observation(self) -> TensorDict:
#         """
#         Creates the observation TensorDict for all agents.
#         为所有智能体创建观测 TensorDict。
#         """
#         # Implement relative distance to targets and other UAVs here
#         # 在此处实现相对于目标和其他无人机的相对距离
#         obs = torch.randn((self.engine.num_uavs, 15)) # Placeholder matching tasks.py
#         return TensorDict({"agents": TensorDict({"observation": obs}, batch_size=[self.engine.num_uavs])}, batch_size=[])

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec, 
    UnboundedContinuousTensorSpec, 
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec
)
from .common import Nav3DClass

class Nav3DEnv(EnvBase):
    def __init__(self, task, **kwargs):
        """
        Wrapper that integrates Nav3DClass with BenchMARL (via TorchRL).
        将 Nav3DClass 与 BenchMARL 集成的包装器 (通过 TorchRL)。
        """
        # EnvBase requires device and batch_size (defaulting to empty for single env)
        super().__init__(device=kwargs.get("device", "cpu"), batch_size=[])
        
        self.task_wrapper = task
        self.config = task.config
        
        # Initialize the physical logic from common.py
        # 初始化 common.py 中的物理逻辑
        self.engine = Nav3DClass(self.config)
        
        # Internal state
        self.uav_pos = None
        self.targets = None

        # --- CRITICAL: Define Specs (TorchRL Requirements) ---
        # These tell the system the shape and range of your data
        n_agents = self.engine.num_uavs
        
        # 1. Action Spec (Continuous movement in X, Y, Z)
        # Assuming actions are roughly -1 to 1 (normalized), but unbounded for safety now
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec(shape=(n_agents, 3))
            })
        })

        # 2. Observation Spec (Placeholder size 15 as per your tasks.py)
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(shape=(n_agents, 15))
            })
        })

        # 3. Reward Spec (One scalar reward per agent)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec(shape=(n_agents, 1))
            })
        })
        
        # 4. Done Spec (Boolean flag per agent)
        self.done_spec = CompositeSpec({
            "done": BinaryDiscreteTensorSpec(n_agents, dtype=torch.bool),
            "terminated": BinaryDiscreteTensorSpec(n_agents, dtype=torch.bool),
            "truncated": BinaryDiscreteTensorSpec(n_agents, dtype=torch.bool),
        })

    def _set_seed(self, seed: int):
        """
        Required by EnvBase. Sets the random seed.
        """
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, tensordict: TensorDict = None, **kwargs) -> TensorDict:
        """
        Resets the environment for a new episode.
        重置环境以开启新的回合。
        """
        # Logic to randomize positions within boundaries
        # 在边界内随机初始化位置的逻辑
        # Note: In a real training, you should use self.config limits here
        self.uav_pos = torch.rand((self.engine.num_uavs, 3)) * 100.0 
        self.targets = torch.rand((self.engine.num_uavs, 3)) * 100.0
        
        return self._make_observation()

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Executes one movement step.
        执行一个移动步。
        """
        # Get actions from the input TensorDict
        actions = tensordict["agents", "action"] # [N, 3]
        
        # Update positions: pos = pos + action * speed
        # 更新位置：位置 = 原位置 + 动作 * 速度
        self.uav_pos = self.uav_pos + (actions * self.engine.uav_speed)
        
        # Check collisions and compute rewards using the engine
        # 使用引擎检查碰撞并计算奖励
        b_col = self.engine.check_building_collisions(self.uav_pos)
        
        # Dummy placeholders for other collision types for now
        uav_col = torch.zeros_like(b_col)
        ground_col = torch.zeros_like(b_col)
        
        reward = self.engine.compute_reward(
            self.uav_pos, self.targets, b_col, uav_col, ground_col
        )
        
        # Generate next observation
        out = self._make_observation()
        
        # Set Reward
        out.set(("next", "agents", "reward"), reward.unsqueeze(-1))
        
        # Set Done/Terminated (All False for now, continuous task)
        done = torch.zeros((self.engine.num_uavs, 1), dtype=torch.bool)
        out.set(("next", "done"), done)
        out.set(("next", "terminated"), done)
        out.set(("next", "truncated"), done)
        
        return out

    def _make_observation(self) -> TensorDict:
        """
        Creates the observation TensorDict for all agents.
        为所有智能体创建观测 TensorDict。
        """
        # Implement relative distance to targets and other UAVs here
        # Placeholder matching tasks.py (size 15)
        obs = torch.randn((self.engine.num_uavs, 15)) 
        
        return TensorDict({
            "agents": TensorDict({
                "observation": obs
            }, batch_size=[self.engine.num_uavs])
        }, batch_size=[])