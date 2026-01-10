"""
Interactive Navigation Script for Tron 1.
Click on the Matplotlib window to send the robot to that location.
"""

import sys
import os

# [环境隔离] 优先使用本地 rsl_rl
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
local_rsl_rl_path = os.path.join(project_root, "rsl_rl")
if local_rsl_rl_path not in sys.path:
    print(f"[INFO] Forcing local rsl_rl import from: {local_rsl_rl_path}")
    sys.path.insert(0, local_rsl_rl_path)

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math

from isaaclab.app import AppLauncher
import cli_args 

# add argparse arguments
parser = argparse.ArgumentParser(description="Click to Navigate with Tron 1.")
parser.add_argument("--task", type=str, default="Isaac-Limx-PF-Normal-Rock-Play-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# [关键] 强制开启相机
args_cli.enable_cameras = True 

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, quat_from_euler_xyz

# [可视化] 引入标记工具
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# [关键] 导入任务包
import bipedal_locomotion 

# 全局变量
TARGET_WORLD_POS = None
camera_data_cache = None

def on_click(event):
    """Matplotlib 点击回调函数: 像素 -> 世界坐标"""
    global TARGET_WORLD_POS, camera_data_cache
    
    if event.xdata is None or event.ydata is None or camera_data_cache is None:
        return

    u, v = int(event.xdata), int(event.ydata)
    print(f"\n[USER] Clicked pixel: ({u}, {v})")

    depth_img = camera_data_cache['depth']
    intrinsic_matrix = camera_data_cache['intrinsics']
    
    # 获取传递过来的机器人状态
    if 'robot_pos' not in camera_data_cache:
        print("[WARN] Robot pose not ready.")
        return
        
    robot_pos = camera_data_cache['robot_pos']
    robot_quat = camera_data_cache['robot_quat']

    # 边界检查
    if v >= depth_img.shape[0] or u >= depth_img.shape[1]: return
    d = depth_img[v, u]
    
    if d <= 0 or d > 100:
        print("[WARN] Clicked on sky or invalid depth! Ignore.")
        return

    print(f"[INFO] Depth at pixel: {d.item():.2f} meters")

    # --- 反投影计算 (基于 Robot Body Frame) ---
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # 1. 计算基础光学坐标
    z_opt = d.item()
    x_opt = (u - cx) * z_opt / fx
    y_opt = (v - cy) * z_opt / fy
    
    # 2. 映射到机器人身体坐标系 (Robot Frame: X=前, Y=左, Z=上)
    # 基于之前验证的逻辑：
    # Robot X (前) = 深度 (z_opt)
    # Robot Y (左) = -图像右 (-x_opt)
    # Robot Z (上) = -图像下 (-y_opt)
    p_robot_flat = torch.tensor([z_opt, -x_opt, -y_opt], device=robot_pos.device)
    
    # 3. 补偿相机安装角度 (Pitch 25度低头)
    # 构造一个 25 度的 Pitch 旋转 (绕 Y 轴)
    pitch_angle = 25.0 * math.pi / 180.0
    q_mount = quat_from_euler_xyz(
        torch.tensor(0.0, device=robot_pos.device),
        torch.tensor(pitch_angle, device=robot_pos.device),
        torch.tensor(0.0, device=robot_pos.device)
    )
    # 旋转向量
    p_robot_tilted = quat_apply(q_mount, p_robot_flat)
    
    # 4. 加上相机相对于机器人的位置偏移 (从 Config 抄来的)
    # pos=(0.35, 0.0, 0.25)
    cam_offset = torch.tensor([0.35, 0.0, 0.25], device=robot_pos.device)
    p_robot_final = p_robot_tilted + cam_offset

    # 5. 转世界坐标
    # 应用机器人当前的旋转 + 机器人当前的位置
    p_world_direction = quat_apply(robot_quat, p_robot_final)
    TARGET_WORLD_POS = robot_pos + p_world_direction
    
    # [必杀技] 强制地面修正 (防止点到地底下或天上)
    TARGET_WORLD_POS[2] = 0.05 
    
    print(f"[NAV] New Target Set: {TARGET_WORLD_POS.cpu().numpy()}")


def main():
    global camera_data_cache, TARGET_WORLD_POS
    
    # 1. 配置解析
    env_cfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path

    # 2. 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 3. 加载策略
    print(f"[INFO]: Loading model: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # 4. [可视化] 初始化目标点标记 (红球)
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    marker_cfg.prim_path = "/Visuals/TargetMarker"
    goal_marker = VisualizationMarkers(marker_cfg)

    # 5. 初始化 Matplotlib
    print("[INFO] Initializing Visualizer...")
    plt.ion()
    fig, ax = plt.subplots()
    img_plot = ax.imshow(np.zeros((480, 640, 3)).astype(np.uint8))
    plt.title("Click to Navigate (RGB Camera View)")
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=False)

    # 6. 初始观测
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    if obs_history is None: obs_history = obs_dict["observations"].get("obsHistory")
    if obs_history is not None: obs_history = obs_history.flatten(start_dim=1)
    
    commands = obs_dict.get("commands")
    if commands is None: commands = obs_dict["observations"].get("commands")

    print("\n[READY] Simulation Started. Please Click on the matplotlib window!")

    last_debug_time = time.time()

    while simulation_app.is_running():
        with torch.inference_mode():
            # --- 1. 相机更新 ---
            try:
                camera = env.unwrapped.scene.sensors["camera"]
            except KeyError:
                camera = list(env.unwrapped.scene.sensors.values())[0]
            
            # 获取机器人状态 (每一帧都要获取最新的)
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
            robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]

            if "rgb" in camera.data.output:
                rgb_tensor = camera.data.output["rgb"][0]
                depth_tensor = camera.data.output["distance_to_image_plane"][0]
                
                rgb_np = rgb_tensor.cpu().numpy()
                img_plot.set_data(rgb_np)
                fig.canvas.flush_events()
                
                camera_data_cache = {
                    'depth': depth_tensor,
                    'intrinsics': camera.data.intrinsic_matrices[0],
                    'robot_pos': robot_pos,   # [关键] 存入机器人位置
                    'robot_quat': robot_quat  # [关键] 存入机器人姿态
                }

            # Debug Print (每5秒打印一次坐标)
            if time.time() - last_debug_time > 5.0:
                pos_np = robot_pos.cpu().numpy()
                print(f"[DEBUG] Robot Position: X={pos_np[0]:.2f}, Y={pos_np[1]:.2f}, Z={pos_np[2]:.2f}")
                last_debug_time = time.time()

            # --- 2. 导航控制 ---
            cmd_vel_x = 0.0
            cmd_ang_z = 0.0
            
            if TARGET_WORLD_POS is not None:
                # [可视化] 画出红点
                goal_marker.visualize(
                    translations=TARGET_WORLD_POS.unsqueeze(0),
                    scales=torch.tensor([[0.2, 0.2, 0.2]], device=env.device)
                )

                target_vec = TARGET_WORLD_POS - robot_pos
                dist = torch.norm(target_vec[:2]) # 水平距离
                
                if dist < 0.3:
                    print("[NAV] Reached Target! Active Idling.")
                    TARGET_WORLD_POS = None
                    # 隐藏红点
                    goal_marker.visualize(translations=torch.tensor([[0,0,-10.0]], device=env.device))
                else:
                    target_local = quat_rotate_inverse(robot_quat, target_vec)
                    
                    # [关键修正] 速度控制逻辑
                    # 距离越远越快，但要有上限
                    # 距离 * 0.8 是增益，clamp 限制在 [-1.0, 1.0] 之间
                    cmd_vel_x = torch.clamp(8 * target_local[0], -12.0, 12.0)
                    
                    # 转向控制
                    yaw_error = torch.atan2(target_local[1], target_local[0])
                    cmd_ang_z = torch.clamp(2.0 * yaw_error, -1.0, 1.0)
                    
                    # [新增] 如果角度偏差太大(>45度)，先原地转，不要硬冲
                    if abs(yaw_error) > 0.8:
                        cmd_vel_x = 0.0 

            # --- 3. 注入指令 & 同步箭头可视化 ---
            
            # [关键] 指令放大 (Command Boosting)
            # 因为 Sim-to-Sim 存在动力不足问题，我们需要放大指令让 Policy 动起来
            # 这里给 5 倍增益 (根据你之前的反馈)
            COMMAND_GAIN = 1.0 
            
            nav_commands = torch.zeros((env.num_envs, 3), device=env.device)
            nav_commands[:, 0] = cmd_vel_x
            nav_commands[:, 1] = 0.0
            nav_commands[:, 2] = cmd_ang_z

            # 给环境显示用的 (保持真实值)
            try:
                env.unwrapped.command_manager.get_term("base_velocity").command[:] = nav_commands
            except:
                pass

            # 给 Policy 的 (放大版)
            policy_commands = nav_commands * COMMAND_GAIN
            
            if commands is not None:
                commands[:] = policy_commands
            
            # --- 4. RL 推理 ---
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            
            # Step
            obs_dict, _, _, _ = env.step(actions)
            
            # 更新 Obs & Commands
            obs = obs_dict["policy"]
            
            obs_history = obs_dict.get("obsHistory")
            if obs_history is None: obs_history = obs_dict["observations"].get("obsHistory")
            if obs_history is not None: obs_history = obs_history.flatten(start_dim=1)
            
            # 保持 commands 引用
            cmds_new = obs_dict.get("commands")
            if cmds_new is None: cmds_new = obs_dict["observations"].get("commands")
            if cmds_new is not None: commands = cmds_new

    env.close()

if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
    simulation_app.close()