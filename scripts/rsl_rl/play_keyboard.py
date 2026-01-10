"""Script to play a checkpoint with Arrow Keys Control and Dedicated Follow Camera."""

import sys
import os

# [环境隔离]
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
local_rsl_rl_path = os.path.join(project_root, "rsl_rl")
if local_rsl_rl_path not in sys.path:
    print(f"[INFO] Forcing local rsl_rl import from: {local_rsl_rl_path}")
    sys.path.insert(0, local_rsl_rl_path)

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher
import cli_args 

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard Control for Tron 1.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of video.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制开启渲染
args_cli.enable_cameras = True 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# Import after AppLauncher
# ==============================================================================
import carb.input
import omni.appwindow
import omni.kit.viewport.utility
import omni.usd
from pxr import Gf, UsdGeom, Usd

import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply
import bipedal_locomotion

# ==============================================================================
# 相机管理类 (创建专用跟随相机)
# ==============================================================================
class FollowCamera:
    def __init__(self, camera_path="/World/FollowCam"):
        self.camera_path = camera_path
        self.stage = omni.usd.get_context().get_stage()
        
        # 1. 创建一个新的相机 Prim，如果不存在的话
        if not self.stage.GetPrimAtPath(camera_path).IsValid():
            self.camera_prim = UsdGeom.Camera.Define(self.stage, camera_path)
            print(f"[INFO] Created dedicated follow camera at: {camera_path}")
        else:
            self.camera_prim = UsdGeom.Camera(self.stage.GetPrimAtPath(camera_path))

        # 2. 强制切换 Viewport 到这个相机
        self.viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if self.viewport_api:
            self.viewport_api.set_active_camera(camera_path)
            print(f"[INFO] Switched viewport to: {camera_path}")
        else:
            print("[WARN] Could not find active viewport!")

    def update(self, eye, target):
        """更新相机位置和朝向"""
        if not self.camera_prim: return

        # 使用 USD Gf 库计算 LookAt 矩阵
        # Isaac Sim 使用 Y-Up (Stage) 或 Z-Up (Physics)
        # 通常 Gf.Vec3d 都是标准数学向量
        eye_vec = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
        target_vec = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
        up_vec = Gf.Vec3d(0, 0, 1) # Z轴向上
        
        # 计算视图矩阵 (World -> Camera)
        # 注意: SetLookAt 生成的是 View Matrix (把世界搬到相机前)
        # 我们需要的是 Camera 的 Transform Matrix (相机在世界里的位置)
        # 所以需要求逆 (GetInverse)
        view_mtx = Gf.Matrix4d().SetLookAt(eye_vec, target_vec, up_vec)
        cam_mtx = view_mtx.GetInverse()
        
        # 写入 USD
        xformable = UsdGeom.Xformable(self.camera_prim)
        xformable.ClearXformOpOrder()
        xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble)
        xform_op.Set(cam_mtx)

# ==============================================================================
# 键盘控制器 (方向键版)
# ==============================================================================
# ==============================================================================
# 键盘控制器 (WASD + 方向键 混合支持)
# ==============================================================================
class KeyboardController:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_yaw = 0.0
        self.selected_env = 0
        
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        
        print("\n" + "=" * 60)
        print("🎮 Keyboard Control Mode Enabled")
        print("------------------------------------------------------------")
        print("UP / W    : Forward")
        print("DOWN / S  : Backward")
        print("LEFT / A  : Turn Left")
        print("RIGHT / D : Turn Right")
        print("Q / E     : Strafe Left / Right")
        print("TAB       : Switch Robot")
        print("============================================================\n")

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.TAB:
                self.selected_env = (self.selected_env + 1) % self.num_envs
                print(f"[INFO] Switched to Robot in Env: {self.selected_env}")
            elif event.input == carb.input.KeyboardInput.R:
                self.vel_x, self.vel_y, self.vel_yaw = 0.0, 0.0, 0.0
        return True

    def update_commands(self):
        # [修改] 加大速度，防止机器人“懒得动”
        # 如果你的 Policy 是在平地训练的，可能需要 1.0 甚至更大才能在月球上跑起来
        speed_linear = 12.0  
        speed_angular = 1.5 
        
        # X Axis: 支持 W 或 UP
        if (self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.W) or 
            self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.UP)):
            self.vel_x = speed_linear
        elif (self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.S) or 
              self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.DOWN)):
            self.vel_x = -speed_linear
        else:
            self.vel_x = 0.0

        # Y Axis: Q/E
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.Q):
            self.vel_y = speed_linear
        elif self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.E):
            self.vel_y = -speed_linear
        else:
            self.vel_y = 0.0

        # Yaw: 支持 A/D 或 LEFT/RIGHT
        if (self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.A) or 
            self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.LEFT)):
            self.vel_yaw = speed_angular
        elif (self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.D) or 
              self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.RIGHT)):
            self.vel_yaw = -speed_angular
        else:
            self.vel_yaw = 0.0
            
        # [调试] 实时打印指令，看看按键有没有生效
        print(f"\r[CMD] X: {self.vel_x:.2f} Y: {self.vel_y:.2f} Yaw: {self.vel_yaw:.2f}", end="")
            
        return self.vel_x, self.vel_y, self.vel_yaw

# ==============================================================================
# Main
# ==============================================================================

def main():
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
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # 初始化控制器和相机
    keyboard = KeyboardController(env.num_envs)
    follow_cam = FollowCamera("/World/FollowCam")

    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    if obs_history is None: obs_history = obs_dict["observations"].get("obsHistory")
    if obs_history is not None: obs_history = obs_history.flatten(start_dim=1)
    
    commands = obs_dict.get("commands")
    if commands is None: commands = obs_dict["observations"].get("commands")

    while simulation_app.is_running():
        with torch.inference_mode():
            
            # 1. 键盘输入
            vx, vy, vyaw = keyboard.update_commands()
            
            # 2. 注入指令
            commands[:, 0] = vx
            commands[:, 1] = vy
            commands[:, 2] = vyaw
            
            try:
                env.unwrapped.command_manager.get_term("base_velocity").command[:] = commands
            except:
                pass

            # 3. 相机跟随
            robot_idx = keyboard.selected_env
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[robot_idx]
            robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[robot_idx]
            
            # 设定跟随位置: 屁股后 2米, 高度 1.5米
            # 如果觉得不够远/高，改这里的 [-2.0, 0.0, 1.5]
            camera_offset_local = torch.tensor([-5.0, 0.0, 1.5], device=env.device)
            camera_offset_world = quat_apply(robot_quat, camera_offset_local)
            
            camera_eye = robot_pos + camera_offset_world
            camera_target = robot_pos + torch.tensor([0.0, 0.0, 0.0], device=env.device) 
            
            follow_cam.update(camera_eye.cpu().numpy(), camera_target.cpu().numpy())

            # 4. RL 推理
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