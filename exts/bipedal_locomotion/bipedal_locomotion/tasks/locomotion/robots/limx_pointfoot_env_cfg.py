import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF.terrains_cfg import (
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
# Pointfoot Base Environment
######################


@configclass
class PFBaseEnvCfg(PFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
class PFBaseEnvCfg_PLAY(PFBaseEnvCfg):
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
# Pointfoot Blind Flat Environment
############################


@configclass
class PFBlindFlatEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# Pointfoot Blind Rough Environment
#############################


@configclass
class PFBlindRoughEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class PFBlindRoughEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
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
# Pointfoot Blind Stairs Environment
##############################


@configclass
class PFBlindStairEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.rewards.rew_lin_vel_xy.weight = 2.0
        self.rewards.rew_ang_vel_z.weight = 1.5
        self.rewards.pen_lin_vel_z.weight = -1.0
        self.rewards.pen_ang_vel_xy.weight = -0.05
        self.rewards.pen_action_rate.weight = -0.01
        self.rewards.pen_flat_orientation.weight = -2.5
        self.rewards.pen_undesired_contacts.weight = -1.0

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class PFBlindStairEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
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
# Pointfoot Stair Environment with height scan
#############################

@configclass
class PFStairEnvCfgv1(PFBaseEnvCfg):
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
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class PFStairEnvCfgv1_PLAY(PFBaseEnvCfg_PLAY):
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
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))










import torch
import math
from dataclasses import MISSING
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFSceneCfg

@configclass
class PFLunarSceneCfg(PFSceneCfg):
    """
    专门为月球导航定制的场景配置。
    继承自 PFSceneCfg，但额外增加了一个 camera 字段。
    """
    camera: CameraCfg | None = MISSING


from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.shapes import SphereCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab.sim import PreviewSurfaceCfg

#####################################
# Normal Gravity Workflow Test Demo
#####################################

@configclass
class PFNormalGravWorkEnvCfg(PFBlindFlatEnvCfg_PLAY):
    scene: PFLunarSceneCfg = PFLunarSceneCfg(num_envs=16, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0 
        
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