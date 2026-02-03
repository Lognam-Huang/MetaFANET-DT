import torch
import json
import os

class Nav3DClass:
    def __init__(self, config):
        """
        Core Environment Logic for 3D UAV Navigation.
        3D 无人机导航核心环境逻辑。
        """
        self.config = config
        self.num_uavs = config.num_uavs
        self.uav_speed = config.uav_speed
        
        # Altitude limits from YAML
        # 来自 YAML 的高度限制
        self.height_min = config.height_min
        self.height_max = config.height_max
        
        # Collision settings
        self.uav_collision_radius = config.uav_collision_radius
        
        # Load building data
        # 加载建筑数据
        with open(config.json_path, 'r') as f:
            self.buildings = json.load(f)
            
        # Convert buildings to tensors for fast GPU/CPU collision checking
        # 将建筑数据转换为张量，以便进行快速的碰撞检查
        self.building_bounds = torch.tensor([
            [b['xmin'], b['ymin'], b['zmin'], b['xmax'], b['ymax'], b['zmax']] 
            for b in self.buildings
        ])

    def check_building_collisions(self, uav_pos):
        """
        Checks if any UAV is inside a building box using AABB logic.
        使用 AABB 逻辑检查无人机是否处于建筑盒内。
        uav_pos: [num_uavs, 3]
        returns: [num_uavs] boolean tensor
        """
        # Expand dims for broadcasting: [num_uavs, 1, 3] vs [1, num_buildings, 6]
        pos = uav_pos.unsqueeze(1) 
        bb = self.building_bounds.to(pos.device).unsqueeze(0)
        
        # Check X, Y, Z constraints
        in_x = (pos[..., 0] >= bb[..., 0]) & (pos[..., 0] <= bb[..., 3])
        in_y = (pos[..., 1] >= bb[..., 1]) & (pos[..., 1] <= bb[..., 4])
        in_z = (pos[..., 2] >= bb[..., 2]) & (pos[..., 2] <= bb[..., 5])
        
        # A collision occurs if a UAV is inside ALL 3 dimensions of ANY building
        collision_matrix = in_x & in_y & in_z
        return collision_matrix.any(dim=1)

    def compute_reward(self, uav_pos, targets, collisions, uav_collisions, reached_target):
        """
        Calculates the bilingual reward logic including the new step penalty.
        计算包含新步数惩罚在内的奖励逻辑。
        """
        # 1. Constant Step Penalty (Encourages speed)
        # 1. 固定步数惩罚（鼓励尽快完成）
        reward = torch.full((self.num_uavs,), self.config.reward_step, device=uav_pos.device)
        
        # 2. Success Bonus
        # 2. 到达目标的奖励
        reward += reached_target.float() * self.config.reward_goal
        
        # 3. Building Collision Penalty
        # 3. 撞击建筑的惩罚
        reward += collisions.float() * self.config.reward_collision
        
        # 4. UAV-UAV Collision Penalty
        # 4. 无人机间碰撞的惩罚
        reward += uav_collisions.float() * self.config.reward_uav_collision
        
        return reward