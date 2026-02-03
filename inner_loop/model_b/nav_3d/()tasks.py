# # # from benchmarl.environments.common import Task
# # # from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec

# # # class RaleighCityTask(Task):
# # #     """
# # #     BenchMARL Task wrapper for the Raleigh 3D Environment.
# # #     Raleigh 3D 环境的 BenchMARL 任务包装器。
# # #     """
# # #     def __init__(self, **kwargs):
# # #         super().__init__()
# # #         self.name = "raleigh_city" # Matches YAML filename
# # #         # from .env import Nav3DEnv
# # #         # self._env_fun = Nav3DEnv

# # #     @property
# # #     def observation_spec(self):
# # #         """
# # #         Defines what each agent observes.
# # #         定义每个智能体的观测空间。
# # #         """
# # #         return CompositeSpec({
# # #             "observation": UnboundedContinuousTensorSpec(shape=(15,)) 
# # #             # Example: self_pos(3), target_rel(3), 3 peers(9) = 15
# # #         })

# # #     @property
# # #     def action_spec(self):
# # #         """
# # #         Defines the 3D velocity control (-1.0 to 1.0).
# # #         定义 3D 速度控制（-1.0 到 1.0）。
# # #         """
# # #         return BoundedTensorSpec(
# # #             low=-1.0, high=1.0, shape=(3,)
# # #         )

# # import torch
# # from benchmarl.environments import Task
# # from benchmarl.environments import task_config_registry
# # from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec, BinaryDiscreteTensorSpec

# # class RaleighCityTask(Task):
# #     # --- CRITICAL: REGISTRY KEY ---
# #     # This string MUST match the 'task=...' argument you use in the command line
# #     RALEIGH_CITY = "nav3d/raleigh_city"

# #     def get_env_fun(self, config):
# #         """
# #         Returns a function that creates the environment.
# #         TorchRL needs a callable (lambda) that returns the env instance.
# #         """
# #         # Import env here to avoid circular import issues during initialization
# #         from .env import Nav3DEnv
        
# #         # Pass the task instance (self) and the config dictionary to the Env
# #         return lambda: Nav3DEnv(task=self, **config)

# #     def supports_continuous_actions(self) -> bool:
# #         return True

# #     def supports_discrete_actions(self) -> bool:
# #         return False

# #     def supports_visual_observations(self) -> bool:
# #         return False

# #     @property
# #     def observation_spec(self):
# #         """
# #         Defines the observation shape (matches env.py).
# #         """
# #         return CompositeSpec({
# #             "agents": CompositeSpec({
# #                 "observation": UnboundedContinuousTensorSpec(shape=(15,)) 
# #             })
# #         })

# #     @property
# #     def action_spec(self):
# #         """
# #         Defines the action shape (matches env.py).
# #         """
# #         return CompositeSpec({
# #             "agents": CompositeSpec({
# #                 "action": BoundedTensorSpec(low=-1.0, high=1.0, shape=(3,))
# #             })
# #         })

# #     @property
# #     def state_spec(self):
# #         """
# #         Optional: Global state for critics (can be same as observation for now).
# #         """
# #         return CompositeSpec({
# #             "agents": CompositeSpec({
# #                 "observation": UnboundedContinuousTensorSpec(shape=(15,))
# #             })
# #         })

# #     @property
# #     def reward_spec(self):
# #         return UnboundedContinuousTensorSpec(shape=(1,))

# # # --- FORCE REGISTRATION ---
# # # This manually inserts your class into BenchMARL's registry
# # # ensuring the KeyError is resolved.
# # task_config_registry["nav3d/raleigh_city"] = RaleighCityTask

# import torch
# from benchmarl.environments import Task
# from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec, BinaryDiscreteTensorSpec

# class RaleighCityTask(Task):
#     # # === ADD THIS INIT METHOD ===
#     # def __init__(self):
#     #     super().__init__()
#     #     self.name = "raleigh_city"
#     # # ============================
    
#     # This string matches your config path
#     RALEIGH_CITY = "nav3d/raleigh_city"

#     def get_env_fun(self, config):
#         """
#         Returns a function that creates the environment.
#         """
#         # Import env here to avoid circular import issues during initialization
#         from .env import Nav3DEnv
#         return lambda: Nav3DEnv(task=self, **config)

#     def supports_continuous_actions(self) -> bool:
#         return True

#     def supports_discrete_actions(self) -> bool:
#         return False

#     def supports_visual_observations(self) -> bool:
#         return False

#     @property
#     def observation_spec(self):
#         return CompositeSpec({
#             "agents": CompositeSpec({
#                 "observation": UnboundedContinuousTensorSpec(shape=(15,)) 
#             })
#         })

#     @property
#     def action_spec(self):
#         return CompositeSpec({
#             "agents": CompositeSpec({
#                 "action": BoundedTensorSpec(low=-1.0, high=1.0, shape=(3,))
#             })
#         })

#     @property
#     def state_spec(self):
#         return CompositeSpec({
#             "agents": CompositeSpec({
#                 "observation": UnboundedContinuousTensorSpec(shape=(15,))
#             })
#         })

#     @property
#     def reward_spec(self):
#         return UnboundedContinuousTensorSpec(shape=(1,))

import torch
# from benchmarl.environments import Task, TaskConfig
# Import from .common to avoid circular dependency with benchmarl.environments
from benchmarl.environments.common import Task, TaskConfig
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec, BinaryDiscreteTensorSpec
from dataclasses import dataclass

# =============================================================================
# 1. THE WORKER (TaskConfig)
#    This class holds all the logic, specs, and environment creation code.
# =============================================================================
@dataclass
class RaleighCityTaskConfig(TaskConfig):
    
    # We move get_env_fun HERE
    def get_env_fun(self, num_envs=1, continuous_actions=True, seed=None, device=None):
        """
        Returns a function that creates the environment.
        """
        # Import env here to avoid circular imports
        from .env import Nav3DEnv
        
        # 'self.config' contains the parameters loaded from hydra (num_uavs, etc.)
        return lambda: Nav3DEnv(task=self, **self.config)

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def supports_visual_observations(self) -> bool:
        return False

    @property
    def observation_spec(self):
        return CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(shape=(15,)) 
            })
        })

    @property
    def action_spec(self):
        return CompositeSpec({
            "agents": CompositeSpec({
                "action": BoundedTensorSpec(low=-1.0, high=1.0, shape=(3,))
            })
        })

    @property
    def state_spec(self):
        return CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(shape=(15,))
            })
        })

    @property
    def reward_spec(self):
        return UnboundedContinuousTensorSpec(shape=(1,))


# =============================================================================
# 2. THE LABEL (Enum)
#    This is what you register. It points to the Worker above.
# =============================================================================
class RaleighCityTask(Task):
    RALEIGH_CITY = "nav3d/raleigh_city"

    def associated_class(self):
        # This tells BenchMARL: "When this task is selected, use this Config class"
        return RaleighCityTaskConfig