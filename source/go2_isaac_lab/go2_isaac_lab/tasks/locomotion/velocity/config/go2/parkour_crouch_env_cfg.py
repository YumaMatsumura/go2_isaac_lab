import isaaclab.terrains as terrain_gen
from go2_isaac_lab.assets.go2 import GO2_CFG
from go2_isaac_lab.tasks.locomotion.velocity import mdp
from go2_isaac_lab.tasks.locomotion.velocity.go2_isaac_lab_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    RewardsCfg,
    TerminationsCfg,
)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .parkour_env_cfg import Go2ParkourObservationsCfg


@configclass
class Go2ParkourCrouchSceneCfg(MySceneCfg):

    def __post_init__(self):
        super().__post_init__()

        # set the robot as Go2
        self.robot = GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=GO2_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.27),
                joint_pos={
                    ".*R_hip_joint": -0.15,
                    ".*L_hip_joint": 0.15,
                    "F[L,R]_thigh_joint": 1.2,
                    "R[L,R]_thigh_joint": 1.3,
                    ".*_calf_joint": -2.0,
                },
            ),
        )

        # terrain parameter settings
        self.terrain.max_init_terrain_level = 5
        self.terrain.terrain_generator.size = (8.0, 8.0)
        self.terrain.terrain_generator.border_width = 1.0
        self.terrain.terrain_generator.num_rows = 10  # 難易度の数
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.05
        self.terrain.terrain_generator.vertical_scale = 0.005
        self.terrain.terrain_generator.slope_threshold = 0.75
        self.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.terrain.terrain_generator.curriculum = True
        self.terrain.terrain_generator.use_cache = False
        self.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.1
        self.terrain.terrain_generator.sub_terrains["random_rough"] = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        )
        self.terrain.terrain_generator.sub_terrains["crouch"] = mdp.CrouchTerrainCfg(
            proportion=0.7,
            inner_width=1.0,
            ceil_width=2.0,
            ceil_height_range=(0.1, 0.5),
        )


@configclass
class Go2ParkourCrouchCommandsCfg(CommandsCfg):

    def __post_init__(self):
        super().__post_init__()

        self.base_velocity.heading_command = False
        self.base_velocity.ranges.lin_vel_x = (0.0, 0.15)
        self.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.base_velocity.limit_ranges.lin_vel_x = (-0.3, 0.3)
        self.base_velocity.limit_ranges.lin_vel_y = (-0.3, 0.3)
        self.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class Go2ParkourCrouchActionsCfg(ActionsCfg):

    def __post_init__(self):
        super().__post_init__()

        self.joint_pos.use_default_offset = False
        self.joint_pos.offset = {
            ".*R_hip_joint": -0.15,
            ".*L_hip_joint": 0.15,
            "F[L,R]_thigh_joint": 1.2,
            "R[L,R]_thigh_joint": 1.3,
            ".*_calf_joint": -2.0,
        }


@configclass
class Go2ParkourCrouchRewardsCfg(RewardsCfg):
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.2,
        params={
            "period": 0.5,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
            "command_name": "base_velocity",
        },
    )
    crouch_height = RewTerm(
        func=mdp.crouch_base_height,
        weight=1.5,
        params={"target_height": 0.20, "higher_scale": 6.0, "lower_scale": 0.5},
    )
    crouch_posture = RewTerm(
        func=mdp.crouch_joint_posture,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "targets": {
                ".*R_hip_joint": -0.15,
                ".*L_hip_joint": 0.15,
                "F[L,R]_thigh_joint": 1.2,
                "R[L,R]_thigh_joint": 1.3,
                ".*_calf_joint": -2.0,
            },
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.undesired_contacts.params["sensor_cfg"].body_names = ["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]

        # weights
        self.track_lin_vel_xy.weight = 2.0  # 指定された速度で進むと報酬が与えられる
        self.track_ang_vel_z.weight = 0.75  # 指定された角速度で旋回すると報酬が与えられる
        self.joint_vel.weight = -0.001  # 関節を速く動かすとペナルティが与えられる
        self.joint_acc.weight = -2.5e-7  # 関節を急に動かすとペナルティが与えられる
        self.joint_torques.weight = -2e-4  # 関節に大きな力を使いすぎるとペナルティが与えられる
        self.action_rate.weight = -0.1  # 急にアクションを変えるとペナルティが与えられる
        self.dof_pos_limits.weight = -3.0  # 関節が可動範囲ギリギリに近づくとペナルティが与えられる
        self.energy.weight = -1e-5  # 消費電力が多いとペナルティが与えられる
        self.joint_pos.weight = 0.0  # 標準姿勢から大きく崩れるとペナルティが与えられる
        self.feet_air_time.weight = 0.05  # 脚が適切に地面から離れると報酬が与えられる
        self.air_time_variance.weight = -0.3  # 左右のスイング時間がバラバラだとペナルティが与えられる
        self.feet_slide.weight = -0.1  # 脚が接地しているのに滑るとペナルティが与えられる
        self.undesired_contacts.weight = -1.0  # 頭や太ももなどが地面に接触するとペナルティが与えられる
        self.feet_gait.weight = 0.1  # トロットのような歩容だと報酬が与えられる
        self.crouch_height.weight = 1.5  # 目標の高さから遠ざかるとペナルティが与えられる
        self.crouch_posture.weight = -0.02  # しゃがみ基本姿勢から遠ざかるとペナルティが与えられる


@configclass
class Go2ParkourCrouchTerminationsCfg(TerminationsCfg):
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 2.5},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.5})

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2ParkourCrouchEnvCfg(LocomotionVelocityRoughEnvCfg):
    num_envs: int = 4096
    env_spacing: float = 2.5
    scene: Go2ParkourCrouchSceneCfg = Go2ParkourCrouchSceneCfg()
    commands: Go2ParkourCrouchCommandsCfg = Go2ParkourCrouchCommandsCfg()
    actions: Go2ParkourCrouchActionsCfg = Go2ParkourCrouchActionsCfg()
    observations: Go2ParkourObservationsCfg = Go2ParkourObservationsCfg()
    rewards: Go2ParkourCrouchRewardsCfg = Go2ParkourCrouchRewardsCfg()
    terminations: Go2ParkourCrouchTerminationsCfg = Go2ParkourCrouchTerminationsCfg()

    def __post_init__(self):
        self.scene.num_envs = self.num_envs
        self.scene.env_spacing = self.env_spacing

        super().__post_init__()


@configclass
class Go2ParkourCrouchEnvCfg_PLAY(Go2ParkourCrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
