import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def pen_lunar_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold_scale: float = 2.0):
    """
    [Physics-Embedded] 重力归一化接触力约束
    """
    # 1. 获取传感器数据 (GPU)
    sensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_forces_norm = torch.norm(foot_forces, dim=-1)
    total_grf = torch.sum(foot_forces_norm, dim=1)
    
    # 2. 获取质量数据 (可能在 CPU)
    robot_mass_per_link = env.scene["robot"].data.default_mass
    total_robot_mass = torch.sum(robot_mass_per_link, dim=1)
    
    # === [关键修复] ===
    # 强制将质量数据移动到与 contact_forces 相同的设备 (通常是 cuda:0)
    total_robot_mass = total_robot_mass.to(total_grf.device)
    # =================
    
    # 3. 计算物理阈值
    moon_gravity = 1.62
    force_limit = total_robot_mass * moon_gravity * threshold_scale
    
    # 4. 计算惩罚
    # 此时 total_grf 和 force_limit 都在 GPU 上，可以相减了
    excess_force = torch.clamp(total_grf - force_limit, min=0.0)
    
    return excess_force

def pen_lunar_vertical_instability(env: ManagerBasedRLEnv):
    """
    [Physics-Embedded] 垂直动力学约束
    """
    # 这个数据本来就在 GPU 上，通常不需要处理
    base_vel_z = env.scene["robot"].data.root_lin_vel_w[:, 2]
    
    ballistic_penalty = torch.square(torch.clamp(base_vel_z, min=0.0))
    
    return ballistic_penalty