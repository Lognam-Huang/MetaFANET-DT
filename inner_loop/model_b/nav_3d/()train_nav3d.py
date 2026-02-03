import os
import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.experiments import Experiment
from benchmarl.models.mlp import MlpConfig
from nav_3d.tasks import RaleighCityTask

def train():
    # 1. Setup paths
    # 1. 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs/raleigh_city.yaml")
    
    # 2. Instantiate your custom task
    # 2. 实例化你的自定义任务
    # We pass the local yaml to our custom Task class
    task = RaleighCityTask(config_path=config_path)

    # 3. Configure the Algorithm (MAPPO)
    # 3. 配置算法 (MAPPO)
    algorithm_config = MappoConfig(
        share_param_net=True,
        lr=3e-4,
        critic_coef=1.0,
        entropy_coef=0.01
    )

    # 4. Create the Experiment
    # 4. 创建实验
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=MlpConfig(num_cells=[128, 128]),
        seed=0,
        config_whitelist=[config_path], # Important: force BenchMARL to read your YAML
        # Storage location
        save_folder=os.path.join(base_dir, "storage")
    )

    # 5. Run Training
    # 5. 运行训练
    experiment.run()

if __name__ == "__main__":
    train()