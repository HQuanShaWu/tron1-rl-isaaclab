"""
Interactive Navigation Script for Tron 1.
Click on the Matplotlib window to send the robot to that location.
"""

import sys
import os

# use local rsl_rl package
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

args_cli.enable_cameras = True 

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz


from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import bipedal_locomotion 

TARGET_WORLD_POS = None
camera_data_cache = None

def on_click(event):
    global TARGET_WORLD_POS, camera_data_cache
    if event.xdata is None or event.ydata is None or camera_data_cache is None:
        return

    u, v = int(event.xdata), int(event.ydata)
    print(f"\n[USER] Clicked pixel: ({u}, {v})")

    depth_img = camera_data_cache['depth']
    intrinsic_matrix = camera_data_cache['intrinsics']
    
    if 'robot_pos' not in camera_data_cache:
        print("[WARN] Robot pose not ready.")
        return
        
    robot_pos = camera_data_cache['robot_pos']
    robot_quat = camera_data_cache['robot_quat']

    if v >= depth_img.shape[0] or u >= depth_img.shape[1]: return
    d = depth_img[v, u]
    if d <= 0 or d > 100:
        print("[WARN] Clicked on sky or invalid depth! Ignore.")
        return

    print(f"[INFO] Depth at pixel: {d.item():.2f} meters")

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    z_opt = d.item()
    x_opt = (u - cx) * z_opt / fx
    y_opt = (v - cy) * z_opt / fy
    
    p_robot_flat = torch.tensor([z_opt, -x_opt, -y_opt], device=robot_pos.device)
    pitch_angle = 20.0 * math.pi / 180.0
    q_mount = quat_from_euler_xyz(
        torch.tensor(0.0, device=robot_pos.device),
        torch.tensor(pitch_angle, device=robot_pos.device),
        torch.tensor(0.0, device=robot_pos.device)
    )

    p_robot_tilted = quat_apply(q_mount, p_robot_flat)
    cam_offset = torch.tensor([0.35, 0.0, 0.25], device=robot_pos.device)
    p_robot_final = p_robot_tilted + cam_offset
    p_world_direction = quat_apply(robot_quat, p_robot_final)
    TARGET_WORLD_POS = robot_pos + p_world_direction
    TARGET_WORLD_POS[2] = 0.05 
    
    print(f"[NAV] New Target Set: {TARGET_WORLD_POS.cpu().numpy()}")


def main():
    global camera_data_cache, TARGET_WORLD_POS
    
    # parse args
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    marker_cfg.prim_path = "/Visuals/TargetMarker"
    goal_marker = VisualizationMarkers(marker_cfg)

    print("[INFO] Initializing Visualizer...")
    plt.ion()
    fig, ax = plt.subplots()
    img_plot = ax.imshow(np.zeros((480, 640, 3)).astype(np.uint8))
    plt.title("Click to Navigate (RGB Camera View)")
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=False)

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
            try:
                camera = env.unwrapped.scene.sensors["camera"]
            except KeyError:
                camera = list(env.unwrapped.scene.sensors.values())[0]
            
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
                    'robot_pos': robot_pos,  
                    'robot_quat': robot_quat 
                }

            if time.time() - last_debug_time > 5.0:
                pos_np = robot_pos.cpu().numpy()
                print(f"[DEBUG] Robot Position: X={pos_np[0]:.2f}, Y={pos_np[1]:.2f}, Z={pos_np[2]:.2f}")
                last_debug_time = time.time()

            cmd_vel_x = 0.0
            cmd_ang_z = 0.0
            
            if TARGET_WORLD_POS is not None:
                goal_marker.visualize(
                    translations=TARGET_WORLD_POS.unsqueeze(0),
                    scales=torch.tensor([[0.2, 0.2, 0.2]], device=env.device)
                )

                target_vec = TARGET_WORLD_POS - robot_pos
                dist = torch.norm(target_vec[:2])
                
                if dist < 0.3:
                    print("[NAV] Reached Target! Active Idling.")
                    TARGET_WORLD_POS = None
                    goal_marker.visualize(translations=torch.tensor([[0,0,-10.0]], device=env.device))
                else:
                    target_local = quat_apply_inverse(robot_quat, target_vec)
                    
                    cmd_vel_x = torch.clamp(8 * target_local[0], -12.0, 12.0)
                    
                    yaw_error = torch.atan2(target_local[1], target_local[0])
                    cmd_ang_z = torch.clamp(2.0 * yaw_error, -1.0, 1.0)
                    
                    if abs(yaw_error) > 0.8:
                        cmd_vel_x = 0.0 

            COMMAND_GAIN = 1.0 
            
            nav_commands = torch.zeros((env.num_envs, 3), device=env.device)
            nav_commands[:, 0] = cmd_vel_x
            nav_commands[:, 1] = 0.0
            nav_commands[:, 2] = cmd_ang_z

            try:
                env.unwrapped.command_manager.get_term("base_velocity").command[:] = nav_commands
            except:
                pass

            policy_commands = nav_commands * COMMAND_GAIN
            
            if commands is not None:
                commands[:] = policy_commands
            
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            
            obs_dict, _, _, _ = env.step(actions)
            
            obs = obs_dict["policy"]
            
            obs_history = obs_dict.get("obsHistory")
            if obs_history is None: obs_history = obs_dict["observations"].get("obsHistory")
            if obs_history is not None: obs_history = obs_history.flatten(start_dim=1)
            
            cmds_new = obs_dict.get("commands")
            if cmds_new is None: cmds_new = obs_dict["observations"].get("commands")
            if cmds_new is not None: commands = cmds_new

    env.close()

if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
    simulation_app.close()