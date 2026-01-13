import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.WF.limx_base_env_cfg import WFEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

from isaaclab.sensors import RayCasterCfg, patterns
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg


######################
# Wheelfoot Base Environment
######################


@configclass
class WFBaseEnvCfg(WFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = WHEELFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        
        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class WFBaseEnvCfg_PLAY(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# Wheelfoot Blind Flat Environment
############################


@configclass
class WFBlindFlatEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class WFBlindFlatEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# Wheelfoot Blind Rough Environment
#############################


@configclass
class WFBlindRoughEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class WFBlindRoughEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG
        

##############################
# Wheelfoot Blind Stairs Environment
##############################

@configclass
class WFBlindStairEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class WFBlindStairEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))
        
        
#############################
# Wheelfoot Flat Environment
#############################

@configclass
class WFFlatEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None

@configclass
class WFFlatEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None
        
        
#############################
# Wheelfoot Rough Environment
#############################

@configclass
class WFRoughEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class WFRoughEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG



        
        
##############################
# Wheelfoot Blind Stairs Environment
##############################


@configclass
class WFStairEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class WFStairEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))
        









#############################
# Wheelfoot Low Flat Environment
#############################
import torch
from dataclasses import MISSING
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFSceneCfg
@configclass
class PFLunarSceneCfg(PFSceneCfg):
    camera: CameraCfg | None = MISSING


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import bipedal_locomotion.tasks.locomotion.my_rewards as my_rewards 

@configclass
class WFLowFlatEnvCfg(WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.sim.gravity = (0.0, 0.0, -1.62)
        
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0) 
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        self.rewards.lunar_vertical_damping = RewTerm(
            func=my_rewards.pen_lunar_vertical_instability,
            weight=-2.0, 
        )
        
        if hasattr(self.rewards, "rew_lin_vel_xy"): 
            self.rewards.rew_lin_vel_xy.weight = 6.0 # from 3.0
            
        if hasattr(self.rewards, "pen_base_height"):
            self.rewards.pen_base_height.weight = -20.0
            self.rewards.pen_base_height.params["target_height"] = 0.60 # from 0.80
            
        if hasattr(self.rewards, "keep_balance"):
            self.rewards.keep_balance.weight = 0.5
            
        if hasattr(self.rewards, "pen_joint_vel_wheel_l2"):
            self.rewards.pen_joint_vel_wheel_l2.weight = -1.0e-4 # from -5e-3


from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.shapes import SphereCfg
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

@configclass
class WFLowFlatEnvCfg_PLAY(WFLowFlatEnvCfg):
    scene: PFLunarSceneCfg = PFLunarSceneCfg(num_envs=1, env_spacing=2.5)
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 3600
        self.scene.num_envs = 1
        self.events.push_robot = None
        self.events.add_base_mass = None

        # Sensor
        # LiDAR
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[4.0, 4.0]), 
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # RGB-D Camera
        q_base = quat_from_euler_xyz(
            torch.tensor(-90.0 * math.pi / 180.0), 
            torch.tensor(0.0), 
            torch.tensor(-90.0 * math.pi / 180.0)
        )
        q_pitch = quat_from_euler_xyz(
            torch.tensor(-20.0 * math.pi / 180.0),
            torch.tensor(0.0), 
            torch.tensor(0.0)
        )
        final_rot = quat_mul(q_base, q_pitch).tolist()
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link/front_cam",
            update_period=0.1,
            height=480, width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=PinholeCameraCfg(focal_length=15.0),
            offset=CameraCfg.OffsetCfg(
                pos=(0.35, 0.0, 0.25), 
                rot=final_rot
            ),
        )

        # A Red Ball for Debug
        self.scene.marker_red_ball = AssetBaseCfg(
            prim_path="/World/MarkerRedBall",
            spawn=SphereCfg(
                radius=0.2,
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), 
                rigid_props=None, 
                collision_props=None, 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, 2.0, 0.5)), 
        )

        # A Blue Ball for Debug
        self.scene.marker_blue_ball = AssetBaseCfg(
            prim_path="/World/MarkerBlueBall",
            spawn=SphereCfg(
                radius=0.2,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)), 
                rigid_props=None, 
                collision_props=None, 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-3.0, 0.0, 0.5)), 
        )

        # A Green Ball for Debug
        self.scene.marker_green_ball = AssetBaseCfg(
            prim_path="/World/MarkerGreenBall",
            spawn=SphereCfg(
                radius=0.2,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), 
                rigid_props=None, 
                collision_props=None, 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, -2.0, 0.5)), 
        )



#############################
# Wheelfoot Low Catena Environment
#############################
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import RigidBodyMaterialCfg

@configclass
class WFLowCatenaEnvCfg(WFLowFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path="/home/img/IsaacLab/tron1-rl-isaaclab/exts/bipedal_locomotion/bipedal_locomotion/assets/usd/moon_terrain/catena160.usd",
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0, 
                dynamic_friction=1.0, 
                restitution=0.0
            ),
            debug_vis=False,
        )

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[4.0, 4.0]), 
            debug_vis=False,
            mesh_prim_paths=["/World/ground/terrain/terrain/mesh"], 
        )
        
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.reset_robot_base.params["pose_range"]["z"] = (0.3, 0.4)
        self.scene.env_spacing = 1.5

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 1024 * 1024 
        self.sim.physx.gpu_max_rigid_contact_count = 10 * 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 256 * 1024 * 1024 
        self.sim.physx.gpu_found_lost_pairs_capacity = 10 * 1024 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 10 * 1024 * 1024

        if hasattr(self.rewards, "lunar_contact_limit"):
            self.rewards.lunar_contact_limit.params["threshold_scale"] = 8.0 
            
        if hasattr(self.rewards, "pen_action_rate"):
            self.rewards.pen_action_rate.weight = -0.1
            

@configclass
class WFLowCatenaEnvCfg_PLAY(WFLowCatenaEnvCfg):
    scene: PFLunarSceneCfg = PFLunarSceneCfg(num_envs=1, env_spacing=2.5)
    def __post_init__(self):
        super().__post_init__()        
        self.scene.num_envs = 64
        
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.episode_length_s = 3600
        
        self.events.reset_robot_base.params["pose_range"]["x"] = (5.0, 8.0) 
        self.events.reset_robot_base.params["pose_range"]["y"] = (-5.0, -8.0)

        # LiDAR
        self.scene.height_scanner.debug_vis = True
        # self.scene.height_scanner = None
        # self.scene.camera = None


        # RGB-D Camera
        q_base = quat_from_euler_xyz(
            torch.tensor(-90.0 * math.pi / 180.0), 
            torch.tensor(0.0), 
            torch.tensor(-90.0 * math.pi / 180.0)
        )
        q_pitch = quat_from_euler_xyz(
            torch.tensor(-20.0 * math.pi / 180.0),
            torch.tensor(0.0), 
            torch.tensor(0.0)
        )
        final_rot = quat_mul(q_base, q_pitch).tolist()
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link/front_cam",
            update_period=0.1,
            height=480, width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=PinholeCameraCfg(focal_length=15.0),
            offset=CameraCfg.OffsetCfg(
                pos=(0.35, 0.0, 0.25), 
                rot=final_rot
            ),
        )


        # A Red Ball for Debug
        self.scene.marker_red_ball = AssetBaseCfg(
            prim_path="/World/MarkerRedBall",
            spawn=SphereCfg(
                radius=0.2,
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), 
                rigid_props=None, 
                collision_props=None, 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, -5.0, 0.8)), 
        )

        # A Blue Ball for Debug
        self.scene.marker_blue_ball = AssetBaseCfg(
            prim_path="/World/MarkerBlueBall",
            spawn=SphereCfg(
                radius=0.2,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)), 
                rigid_props=None, 
                collision_props=None, 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -8.0, 0.8)), 
        )

        # A Green Ball for Debug
        self.scene.marker_green_ball = AssetBaseCfg(
            prim_path="/World/MarkerGreenBall",
            spawn=SphereCfg(
                radius=0.2,
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), 
                rigid_props=None, 
                collision_props=None, 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, -8.0, 0.8)), 
        )