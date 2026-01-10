"""
[Body] Isaac Sim Navigation Server.
Listens for commands from VLM Client, captures images, and handles navigation.
Synced with click_and_go.py logic.
"""

import sys
import os
import socket
import json
import select
import cv2
import numpy as np
import torch
import math
import time

# [环境隔离]
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
local_rsl_rl_path = os.path.join(project_root, "rsl_rl")
if local_rsl_rl_path not in sys.path:
    sys.path.insert(0, local_rsl_rl_path)

import argparse
from isaaclab.app import AppLauncher
import cli_args 

parser = argparse.ArgumentParser(description="Tron 1 Navigation Server")
parser.add_argument("--task", type=str, default="Isaac-Limx-PF-Normal-Rock-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint_path", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after Sim Start ---
import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, quat_from_euler_xyz
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import bipedal_locomotion 

# ==============================================================================
# Socket Server Logic
# ==============================================================================
HOST = '127.0.0.1'
PORT = 65432
IMG_SAVE_PATH = os.path.abspath("temp_sim_view.jpg")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()
server_socket.setblocking(False)

print(f"\n[SERVER] Listening on {HOST}:{PORT}...")

# ==============================================================================
# Global State
# ==============================================================================
TARGET_WORLD_POS = None
SNAPSHOT_DATA = None 

def main():
    global TARGET_WORLD_POS, SNAPSHOT_DATA
    
    # 1. Sim Setup
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
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

    # 2. RL Setup
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # 3. Visuals
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    marker_cfg.prim_path = "/Visuals/TargetMarker"
    goal_marker = VisualizationMarkers(marker_cfg)

    # 4. Init Obs (Safe Access)
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    
    try:
        obs_history = obs_dict["obsHistory"]
    except KeyError:
        obs_history = obs_dict["observations"]["obsHistory"]
    obs_history = obs_history.flatten(start_dim=1)
    
    try:
        commands = obs_dict["commands"]
    except KeyError:
        commands = obs_dict["observations"]["commands"]

    print("[SERVER] Simulation running. Waiting for VLM Client...")

    while simulation_app.is_running():
        
        # ==========================================
        # Part A: Handle Network Requests
        # ==========================================
        readable, _, _ = select.select([server_socket], [], [], 0.0) 
        
        if readable:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(4096)
                if data:
                    try:
                        req = json.loads(data.decode('utf-8'))
                        cmd = req.get("cmd")
                        
                        if cmd == "CAPTURE":
                            # 1. 抓拍
                            try:
                                camera = env.unwrapped.scene.sensors["camera"]
                            except:
                                camera = list(env.unwrapped.scene.sensors.values())[0]
                                
                            if "rgb" in camera.data.output:
                                rgb_np = camera.data.output["rgb"][0].cpu().numpy()
                                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(IMG_SAVE_PATH, bgr_np)
                                
                                # [关键] 保存那一瞬间的机器人位姿，用于反投影
                                SNAPSHOT_DATA = {
                                    'depth': camera.data.output["distance_to_image_plane"][0].clone(),
                                    'intrinsics': camera.data.intrinsic_matrices[0].clone(),
                                    # 必须存机器人的位姿，因为坐标系转换是基于机器人的
                                    'robot_pos': env.unwrapped.scene["robot"].data.root_pos_w[0].clone(),
                                    'robot_quat': env.unwrapped.scene["robot"].data.root_quat_w[0].clone()
                                }
                                
                                resp = {"status": "OK", "image_path": IMG_SAVE_PATH}
                                print(f"[SERVER] Snapshot taken. Robot Pos: {SNAPSHOT_DATA['robot_pos'].cpu().numpy()}")
                            else:
                                resp = {"status": "ERR", "msg": "Camera not ready"}
                                
                        elif cmd == "NAVIGATE":
                            if SNAPSHOT_DATA is None:
                                resp = {"status": "ERR", "msg": "No snapshot taken"}
                            else:
                                u, v = req["data"]["u"], req["data"]["v"]
                                print(f"[SERVER] Received Target Pixel: ({u}, {v})")
                                
                                depth_img = SNAPSHOT_DATA['depth']
                                
                                # [新增] 边界检查 + 深度有效性检查
                                if v < depth_img.shape[0] and u < depth_img.shape[1]:
                                    d = depth_img[v, u].item()
                                    
                                    # [关键修复] 过滤无效深度 (天空/无穷远/噪点)
                                    # 假设最远只允许导航到 20 米以外，超过就认为是无效点击
                                    if d <= 0.0 or d > 20.0 or math.isnan(d) or math.isinf(d):
                                        print(f"[WARN] Invalid Depth: {d}. Likely clicked on Sky/Void.")
                                        resp = {"status": "ERR", "msg": "Target is sky or too far"}
                                    else:
                                        # --- 深度有效，继续计算坐标 ---
                                        intr = SNAPSHOT_DATA['intrinsics']
                                        fx, fy = intr[0,0].item(), intr[1,1].item()
                                        cx, cy = intr[0,2].item(), intr[1,2].item()
                                        
                                        # 1. 光学坐标
                                        z_opt = d
                                        x_opt = (u - cx) * z_opt / fx
                                        y_opt = (v - cy) * z_opt / fy
                                        
                                        # 2. 机器人坐标系 (Robot Frame)
                                        # X=前(Depth), Y=左(-Right), Z=上(-Down)
                                        vec_robot = torch.tensor([z_opt, -x_opt, -y_opt], device=env.device)
                                        
                                        # 3. 补偿 Pitch
                                        pitch_angle = 25.0 * math.pi / 180.0
                                        q_mount = quat_from_euler_xyz(
                                            torch.tensor(0.0, device=env.device),
                                            torch.tensor(pitch_angle, device=env.device),
                                            torch.tensor(0.0, device=env.device)
                                        )
                                        vec_robot_tilted = quat_apply(q_mount, vec_robot)
                                        
                                        # 4. 转世界坐标 (加上机器人位姿)
                                        # 注意: 这里的 vec_robot_tilted 已经是相对于 Robot Base 的向量了
                                        # 所以直接用 SNAPSHOT 里的机器人位姿旋转它
                                        p_world_dir = quat_apply(SNAPSHOT_DATA['robot_quat'], vec_robot_tilted)
                                        TARGET_WORLD_POS = SNAPSHOT_DATA['robot_pos'] + p_world_dir
                                        
                                        TARGET_WORLD_POS[2] = 0.05
                                        
                                        resp = {"status": "MOVING"}
                                        print(f"[SERVER] Valid Target: {TARGET_WORLD_POS.cpu().numpy()}")
                                else:
                                    resp = {"status": "ERR", "msg": "Pixel out of bounds"}

                        conn.sendall(json.dumps(resp).encode('utf-8'))
                    except Exception as e:
                        print(f"[SERVER] Error: {e}")

        # ==========================================
        # Part B: Navigation Loop (Real-time Control)
        # ==========================================
        with torch.inference_mode():
            cmd_vel_x = 0.0
            cmd_ang_z = 0.0
            
            if TARGET_WORLD_POS is not None:
                # 可视化红点
                goal_marker.visualize(
                    translations=TARGET_WORLD_POS.unsqueeze(0),
                    scales=torch.tensor([[0.2, 0.2, 0.2]], device=env.device)
                )
                
                # 获取当前实时位姿
                curr_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                curr_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
                
                target_vec = TARGET_WORLD_POS - curr_pos
                dist = torch.norm(target_vec[:2])
                
                if dist < 0.3:
                    print("[SERVER] Reached.")
                    TARGET_WORLD_POS = None
                    goal_marker.visualize(translations=torch.tensor([[0,0,-10.0]], device=env.device))
                else:
                    target_local = quat_rotate_inverse(curr_quat, target_vec)
                    
                    # [关键参数同步] 使用你测试通过的激进参数
                    # 1. 速度增益: 8.0, 限幅: 12.0
                    cmd_vel_x = torch.clamp(8.0 * target_local[0], -12.0, 12.0)
                    
                    # 2. 转向控制: 2.0 * error
                    yaw_error = torch.atan2(target_local[1], target_local[0])
                    cmd_ang_z = torch.clamp(2.0 * yaw_error, -1.0, 1.0)
                    
                    # [可选] 防止原地转弯时速度过大导致漂移，可以加个判断
                    if abs(yaw_error) > 1.0: # 角度大时减速
                         cmd_vel_x *= 0.1

            # 注入指令
            if commands is not None:
                # COMMAND_GAIN = 1.0 (因为上面已经乘了8.0了)
                commands[:, 0] = cmd_vel_x
                commands[:, 1] = 0.0
                commands[:, 2] = cmd_ang_z
            
            # 箭头同步
            try:
                env.unwrapped.command_manager.get_term("base_velocity").command[:] = commands
            except: pass

            # RL 推理
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs_dict, _, _, _ = env.step(actions)
            
            # 更新 Obs
            obs = obs_dict["policy"]
            try:
                obs_history_raw = obs_dict["obsHistory"]
            except KeyError:
                obs_history_raw = obs_dict["observations"]["obsHistory"]
            obs_history = obs_history_raw.flatten(start_dim=1)
            
            # 更新 commands 引用
            cmds_new = None
            try:
                cmds_new = obs_dict["commands"]
            except KeyError:
                try:
                    cmds_new = obs_dict["observations"]["commands"]
                except KeyError: pass     
            if cmds_new is not None: commands = cmds_new

    env.close()

if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
    simulation_app.close()