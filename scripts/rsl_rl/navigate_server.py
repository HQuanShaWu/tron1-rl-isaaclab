"""
[Body] Isaac Sim Navigation Server.
Listens for commands from VLM Client, captures images, and handles navigation.
Updated with Timeout Safety Mechanism.
"""

import sys
import os

# use local rsl_rl package
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
local_rsl_rl_path = os.path.join(project_root, "rsl_rl")
if local_rsl_rl_path not in sys.path:
    sys.path.insert(0, local_rsl_rl_path)

import socket
import json
import select
import cv2
import torch
import math
import argparse
import time

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

import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import bipedal_locomotion 

def get_yaw_from_quat(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw


# Socket Server Logic
HOST = '127.0.0.1'
PORT = 65432
IMG_SAVE_PATH = os.path.abspath("temp_sim_view.jpg")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()
server_socket.setblocking(False)

print(f"\n[SERVER] Listening on {HOST}:{PORT}...")


# Global State
TARGET_WORLD_POS = None
SNAPSHOT_DATA = None 
IS_TURNING_MODE = False
TARGET_YAW = None
ROBOT_STATUS = "IDLE"
TASK_START_TIME = 0.0
TASK_TIMEOUT = 15.0

def main():
    global TARGET_WORLD_POS, SNAPSHOT_DATA, IS_TURNING_MODE, TARGET_YAW, ROBOT_STATUS, TASK_START_TIME

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

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    marker_cfg.prim_path = "/Visuals/TargetMarker"
    goal_marker = VisualizationMarkers(marker_cfg)

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
                            try:
                                camera = env.unwrapped.scene.sensors["camera"]
                            except:
                                camera = list(env.unwrapped.scene.sensors.values())[0]
                            if "rgb" in camera.data.output:
                                rgb_np = camera.data.output["rgb"][0].cpu().numpy()
                                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(IMG_SAVE_PATH, bgr_np)
                                SNAPSHOT_DATA = {
                                    'depth': camera.data.output["distance_to_image_plane"][0].clone(),
                                    'intrinsics': camera.data.intrinsic_matrices[0].clone(),
                                    'robot_pos': env.unwrapped.scene["robot"].data.root_pos_w[0].clone(),
                                    'robot_quat': env.unwrapped.scene["robot"].data.root_quat_w[0].clone()
                                }
                                resp = {"status": "OK", "image_path": IMG_SAVE_PATH}
                                print(f"[SERVER] Snapshot taken. Robot Pos: {SNAPSHOT_DATA['robot_pos'].cpu().numpy()}")
                            else:
                                resp = {"status": "ERR", "msg": "Camera not ready"}
                        elif cmd == "GET_STATUS":
                            resp = {"status": ROBOT_STATUS}

                        elif cmd == "STOP":
                            print("[SERVER] Command: STOP")
                            TARGET_WORLD_POS = None
                            IS_TURNING_MODE = False
                            TARGET_YAW = None
                            goal_marker.visualize(translations=torch.tensor([[0,0,-10.0]], device=env.device))
                            ROBOT_STATUS = "IDLE"
                            resp = {"status": "STOPPED"}

                        elif cmd == "TURN":
                            direction = req["data"]["direction"]
                            angle_deg = req["data"].get("angle", 90.0)
                            
                            print(f"[SERVER] Command: TURN {'Left' if direction > 0 else 'Right'} {angle_deg} deg")
                            
                            curr_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
                            curr_yaw = get_yaw_from_quat(curr_quat)
                            
                            angle_rad = angle_deg * (math.pi / 180.0)
                            target_yaw = curr_yaw + (direction * angle_rad)
                            
                            TARGET_WORLD_POS = None
                            IS_TURNING_MODE = True
                            TARGET_YAW = target_yaw
                            ROBOT_STATUS = "BUSY"
                            TASK_START_TIME = time.time()

                            goal_marker.visualize(translations=torch.tensor([[0,0,-10.0]], device=env.device))
                            resp = {"status": "TURNING"}


                        elif cmd == "NAVIGATE":
                            if SNAPSHOT_DATA is None:
                                resp = {"status": "ERR", "msg": "No snapshot taken"}
                            else:
                                u, v = req["data"]["u"], req["data"]["v"]
                                print(f"[SERVER] Received Target Pixel: ({u}, {v})")
                                
                                depth_img = SNAPSHOT_DATA['depth']
                                
                                if v < depth_img.shape[0] and u < depth_img.shape[1]:
                                    d = depth_img[v, u].item()
                                    
                                    if d <= 0.0 or d > 30.0 or math.isnan(d) or math.isinf(d):
                                        print(f"[WARN] Invalid Depth: {d}. Likely clicked on Sky/Void.")
                                        resp = {"status": "ERR", "msg": "Target is sky or too far"}
                                    else:
                                        intr = SNAPSHOT_DATA['intrinsics']
                                        fx, fy = intr[0,0].item(), intr[1,1].item()
                                        cx, cy = intr[0,2].item(), intr[1,2].item()
                                        
                                        z_opt = d
                                        x_opt = (u - cx) * z_opt / fx
                                        y_opt = (v - cy) * z_opt / fy
                                        
                                        vec_robot = torch.tensor([z_opt, -x_opt, -y_opt], device=env.device)
                                        
                                        pitch_angle = 20.0 * math.pi / 180.0
                                        q_mount = quat_from_euler_xyz(
                                            torch.tensor(0.0, device=env.device),
                                            torch.tensor(pitch_angle, device=env.device),
                                            torch.tensor(0.0, device=env.device)
                                        )
                                        vec_robot_tilted = quat_apply(q_mount, vec_robot)
                                        
                                        p_world_dir = quat_apply(SNAPSHOT_DATA['robot_quat'], vec_robot_tilted)
                                        TARGET_WORLD_POS = SNAPSHOT_DATA['robot_pos'] + p_world_dir
                                        
                                        TARGET_WORLD_POS[2] = 0.05

                                        ROBOT_STATUS = "BUSY"
                                        TASK_START_TIME = time.time()
                                        resp = {"status": "MOVING"}
                                        print(f"[SERVER] Valid Target: {TARGET_WORLD_POS.cpu().numpy()}")
                                else:
                                    resp = {"status": "ERR", "msg": "Pixel out of bounds"}

                        conn.sendall(json.dumps(resp).encode('utf-8'))
                    except Exception as e:
                        print(f"[SERVER] Error: {e}")

        with torch.inference_mode():
            cmd_vel_x = 0.0
            cmd_ang_z = 0.0

            curr_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]

            if ROBOT_STATUS == "BUSY" and (time.time() - TASK_START_TIME > TASK_TIMEOUT):
                print(f"[SERVER] Warning: Task Timeout ({TASK_TIMEOUT}s). Forcing IDLE.")
                ROBOT_STATUS = "IDLE"
                IS_TURNING_MODE = False
                TARGET_WORLD_POS = None
                TARGET_YAW = None
                goal_marker.visualize(translations=torch.tensor([[0,0,-10.0]], device=env.device))

            if IS_TURNING_MODE and TARGET_YAW is not None:
                curr_yaw = get_yaw_from_quat(curr_quat)
                yaw_error = TARGET_YAW - curr_yaw
                
                while yaw_error > math.pi: yaw_error -= 2 * math.pi
                while yaw_error < -math.pi: yaw_error += 2 * math.pi
                
                if abs(yaw_error) < 0.08:
                    if ROBOT_STATUS == "BUSY": 
                        print("[SERVER] Turn Complete. Status -> IDLE")  
                        ROBOT_STATUS = "IDLE" 
                    IS_TURNING_MODE = False
                    TARGET_YAW = None
                    cmd_ang_z = 0.0
                else:
                    cmd_vel_x = 0.0
                    cmd_ang_z = torch.clamp(2.5 * yaw_error, -1.0, 1.0)
            
            elif TARGET_WORLD_POS is not None:
                goal_marker.visualize(
                    translations=TARGET_WORLD_POS.unsqueeze(0),
                    scales=torch.tensor([[0.2, 0.2, 0.2]], device=env.device)
                )
                
                curr_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                target_vec = TARGET_WORLD_POS - curr_pos
                dist = torch.norm(target_vec[:2])
                
                if dist < 0.3:
                    if ROBOT_STATUS == "BUSY":
                        print("[SERVER] Reached Target. Status -> IDLE")
                        ROBOT_STATUS = "IDLE"
                    TARGET_WORLD_POS = None
                    goal_marker.visualize(translations=torch.tensor([[0,0,-10.0]], device=env.device))
                else:
                    target_local = quat_apply_inverse(curr_quat, target_vec)
                    
                    cmd_vel_x = torch.clamp(1.0 * target_local[0], -1.2, 1.2)
                    yaw_error = torch.atan2(target_local[1], target_local[0])
                    cmd_ang_z = torch.clamp(2.0 * yaw_error, -1.0, 1.0)
                    
                    if abs(yaw_error) > 1.0:
                         cmd_vel_x *= 0.1

            if commands is not None:
                commands[:, 0] = cmd_vel_x
                commands[:, 1] = 0.0
                commands[:, 2] = cmd_ang_z

            try:
                env.unwrapped.command_manager.get_term("base_velocity").command[:] = commands
            except: pass

            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs_dict, _, _, _ = env.step(actions)

            obs = obs_dict["policy"]
            try:
                obs_history_raw = obs_dict["obsHistory"]
            except KeyError:
                obs_history_raw = obs_dict["observations"]["obsHistory"]
            obs_history = obs_history_raw.flatten(start_dim=1)

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