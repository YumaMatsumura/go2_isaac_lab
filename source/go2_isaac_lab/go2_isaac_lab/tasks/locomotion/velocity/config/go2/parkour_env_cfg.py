import isaaclab.terrains as terrain_gen
from go2_isaac_lab.assets.go2 import GO2_CFG
from go2_isaac_lab.tasks.locomotion.velocity import mdp
from go2_isaac_lab.tasks.locomotion.velocity.go2_isaac_lab_env_cfg import (
    CommandsCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# ----- Common -----
ALPHA1 = 0.2
ALPHA2 = 0.4
ALPHA3 = 0.6
ALPHAS = (
    ("alpha1", ALPHA1),
    ("alpha2", ALPHA2),
    ("alpha3", ALPHA3),
)
EMA_BASE_CONFIGS = {
    "base_ang_vel": (
        "base_ang_vel",
        mdp.base_ang_vel,
        {},
    ),
    "projected_gravity": (
        "projected_gravity",
        mdp.projected_gravity,
        {},
    ),
    "velocity_commands": (
        "velocity_commands",
        mdp.generated_commands,
        {"command_name": "base_velocity"},
    ),
    "joint_pos_rel": (
        "joint_pos_rel",
        mdp.joint_pos_rel,
        {},
    ),
    "joint_vel_rel": (
        "joint_vel_rel",
        mdp.joint_vel_rel,
        {},
    ),
    "last_action": (
        "last_action",
        mdp.last_action,
        {},
    ),
}


@configclass
class Go2ParkourObservationsCfg(ObservationsCfg):

    @configclass
    class CommonCfg(ObservationsCfg.PolicyCfg):

        def __post_init__(self):
            super().__post_init__()

            # scale observations
            self.base_ang_vel.scale = 0.2
            self.projected_gravity.scale = 1.0
            self.velocity_commands.scale = 1.0
            self.joint_pos_rel.scale = 1.0
            self.joint_vel_rel.scale = 0.05
            self.last_action.scale = 1.0

            # clip observations
            self.base_ang_vel.clip = (-100, 100)
            self.projected_gravity.clip = (-100, 100)
            self.velocity_commands.clip = (-100, 100)
            self.joint_pos_rel.clip = (-100, 100)
            self.joint_vel_rel.clip = (-100, 100)
            self.last_action.clip = (-100, 100)

    @configclass
    class PolicyCfg(CommonCfg):

        def __post_init__(self):
            super().__post_init__()

            # add noise to the observations
            self.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
            self.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
            self.joint_pos_rel.noise = Unoise(n_min=-0.01, n_max=0.01)
            self.joint_vel_rel.noise = Unoise(n_min=-1.5, n_max=1.5)

            # EMA
            for suffix, alpha in ALPHAS:
                for ema_name, (base_name, base_func, base_params) in EMA_BASE_CONFIGS.items():
                    term_name = f"{ema_name}_ema_{suffix}"

                    # ObsTermを生成してこのクラスに生やす
                    ema_term = ObsTerm(
                        func=mdp.ema_single,
                        params={
                            "base_func": base_func,
                            "key": base_name,
                            "alpha": alpha,
                            "base_params": base_params,
                        },
                    )
                    setattr(self, term_name, ema_term)

                    # scale / clip / noise を元のObsTermからコピー
                    base_term = getattr(self, base_name)
                    ema_term.scale = base_term.scale
                    ema_term.clip = base_term.clip
                    ema_term.noise = None

    @configclass
    class CriticCfg(CommonCfg):
        # --- Add more privileged observations ---
        # 速度（3次元）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        # 関節トルク（12次元）
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        # Height Scan（187次元）
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        # Friction Coefficient（1次元）
        friction_coeff = ObsTerm(
            func=mdp.friction_coeff,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mode": "mean",
            },
            clip=(0.3, 1.2),  # EventCfgでのrandomization rangeに合わせる
            scale=1.0,
        )
        # Base Mass（1次元）
        base_mass = ObsTerm(
            func=mdp.body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
            clip=(0.0, 100.0),  # ロボットの質量スケールに合わせる
            scale=1.0 / 30.0,
        )

        def __post_init__(self):
            super().__post_init__()

    policy = PolicyCfg()
    critic = CriticCfg()

    def __post_init__(self):
        return super().__post_init__()


# ----- Stationary -----
@configclass
class Go2ParkourStationarySceneCfg(MySceneCfg):

    def __post_init__(self):
        super().__post_init__()

        # set the robot as Go2
        # self.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # for play
        self.robot = GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=GO2_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.01),
            ),
        )

        # terrain parameter settings
        self.terrain.max_init_terrain_level = 5
        self.terrain.terrain_generator.size = (8.0, 8.0)
        self.terrain.terrain_generator.border_width = 20.0
        self.terrain.terrain_generator.num_rows = 10  # 難易度の数
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.05
        self.terrain.terrain_generator.vertical_scale = 0.005
        self.terrain.terrain_generator.slope_threshold = 0.75
        self.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.terrain.terrain_generator.curriculum = True
        self.terrain.terrain_generator.use_cache = False
        self.terrain.terrain_generator.sub_terrains["flat"].proportion = 1.0


@configclass
class Go2ParkourStationaryCommandsCfg(CommandsCfg):

    def __post_init__(self):
        super().__post_init__()

        self.base_velocity.heading_command = False
        self.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
        self.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)


@configclass
class Go2ParkourStationaryEventCfg(EventCfg):

    def __post_init__(self):
        super().__post_init__()

        # first half (9900 iterations)
        # self.reset_base.params["pose_range"]["z"] = (0.25, 0.35)
        # self.reset_base.params["pose_range"]["roll"] = (-1.2, 1.2)
        # self.reset_base.params["pose_range"]["pitch"] = (-1.2, 1.2)
        # self.reset_robot_joints.params["position_range"] = (-0.5, 0.5)

        # second half (20000 iterations)
        # self.reset_base.params["pose_range"]["z"] = (0.25, 0.35)
        # self.reset_base.params["pose_range"]["roll"] = (-3.14, 3.14)
        # self.reset_base.params["pose_range"]["pitch"] = (-3.14, 3.14)
        # self.reset_robot_joints.params["position_range"] = (-1.0, 1.0)

        # for play
        self.reset_base.params["pose_range"]["z"] = (0.01, 0.01)
        self.reset_base.params["pose_range"]["roll"] = (-3.14, -3.00)
        self.reset_base.params["pose_range"]["pitch"] = (-0.1, 0.1)
        self.reset_robot_joints.params["position_range"] = (-0.1, 0.1)


@configclass
class Go2ParkourStationaryRewardsCfg(RewardsCfg):
    # baseの高さがずれるとペナルティが与えられる
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={
            "target_height": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": None,
        },
    )
    # 倒れているとペナルティが与えられる
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 足が接地しているときに報酬を与える
    feet_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=10.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_foot"],
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.undesired_contacts.params["sensor_cfg"].body_names = ["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]

        # weights
        self.track_lin_vel_xy.weight = 0.0  # 指定された速度で進むと報酬が与えられる
        self.track_ang_vel_z.weight = 0.0  # 指定された角速度で旋回すると報酬が与えられる
        self.joint_vel.weight = -5e-4  # 関節を速く動かすとペナルティが与えられる
        self.joint_acc.weight = -1e-7  # 関節を急に動かすとペナルティが与えられる
        self.joint_torques.weight = -1e-6  # 関節に大きな力を使いすぎるとペナルティが与えられる
        self.action_rate.weight = -0.02  # 急にアクションを変えるとペナルティが与えられる
        self.dof_pos_limits.weight = -2.0  # 関節が可動範囲ギリギリに近づくとペナルティが与えられる
        self.energy.weight = 0.0  # 消費電力が多いとペナルティが与えられる
        self.joint_pos.weight = -0.2  # 標準姿勢から大きく崩れるとペナルティが与えられる
        self.feet_air_time.weight = 0.0  # 脚が適切に地面から離れると報酬が与えられる
        self.air_time_variance.weight = 0.0  # 左右のスイング時間がバラバラだとペナルティが与えられる
        self.feet_slide.weight = -0.05  # 脚が接地しているのに滑るとペナルティが与えられる
        self.undesired_contacts.weight = -5.0  # 頭や太ももなどが地面に接触するとペナルティが与えられる
        self.base_height.weight = -20.0  # baseの高さがずれるとペナルティが与えられる
        self.flat_orientation.weight = -5.0  # 倒れているとペナルティが与えられる
        self.feet_contacts.weight = 5.0  # 脚が地面に接触すると報酬が与えられる


@configclass
class Go2ParkourStationaryTerminationsCfg(TerminationsCfg):

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2ParkourStationaryEnvCfg(LocomotionVelocityRoughEnvCfg):
    num_envs: int = 4096
    env_spacing: float = 2.5
    scene: Go2ParkourStationarySceneCfg = Go2ParkourStationarySceneCfg()
    commands: Go2ParkourStationaryCommandsCfg = Go2ParkourStationaryCommandsCfg()
    observations: Go2ParkourObservationsCfg = Go2ParkourObservationsCfg()
    events: Go2ParkourStationaryEventCfg = Go2ParkourStationaryEventCfg()
    rewards: Go2ParkourStationaryRewardsCfg = Go2ParkourStationaryRewardsCfg()
    # terminations: Go2ParkourStationaryTerminationsCfg = Go2ParkourStationaryTerminationsCfg()

    def __post_init__(self):
        self.scene.num_envs = self.num_envs
        self.scene.env_spacing = self.env_spacing

        super().__post_init__()

        self.episode_length_s = 10.0


@configclass
class Go2ParkourStationaryEnvCfg_PLAY(Go2ParkourStationaryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ----- Climb Up -----
@configclass
class Go2ParkourClimbUpSceneCfg(MySceneCfg):

    def __post_init__(self):
        super().__post_init__()

        # set the robot as Go2
        self.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain parameter settings
        self.terrain.max_init_terrain_level = 5
        self.terrain.terrain_generator.size = (8.0, 8.0)
        self.terrain.terrain_generator.border_width = 20.0
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
            proportion=0.1,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        )
        self.terrain.terrain_generator.sub_terrains["square_pit_power"] = mdp.SquarePitPowerTerrainCfg(
            proportion=0.80,
            pit_edge=1.6,  # 穴の一辺 [m]
            rim_height_range=(0.3, 0.7),  # 外周高さ
            rise_length_range=(1.2, 0.02),  # 崖化
            growth_exponent=2.0,  # 形状（べき指数）
            height_cutoff=1e-4,
            noise_std=0.001,  # 荒れ地ノイズ
        )


@configclass
class Go2ParkourClimbUpRewardsCfg(RewardsCfg):
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.undesired_contacts.params["sensor_cfg"].body_names = ["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]

        # weights
        self.track_lin_vel_xy.weight = 1.5  # 指定された速度で進むと報酬が与えられる
        self.track_ang_vel_z.weight = 0.75  # 指定された角速度で旋回すると報酬が与えられる
        self.joint_vel.weight = -0.001  # 関節を速く動かすとペナルティが与えられる
        self.joint_acc.weight = -2.5e-7  # 関節を急に動かすとペナルティが与えられる
        self.joint_torques.weight = -2e-4  # 関節に大きな力を使いすぎるとペナルティが与えられる
        self.action_rate.weight = -0.1  # 急にアクションを変えるとペナルティが与えられる
        self.dof_pos_limits.weight = -5.0  # 関節が可動範囲ギリギリに近づくとペナルティが与えられる
        self.energy.weight = -2e-5  # 消費電力が多いとペナルティが与えられる
        self.joint_pos.weight = -0.5  # 標準姿勢から大きく崩れるとペナルティが与えられる
        self.feet_air_time.weight = 0.1  # 脚が適切に地面から離れると報酬が与えられる
        self.air_time_variance.weight = -1.0  # 左右のスイング時間がバラバラだとペナルティが与えられる
        self.feet_slide.weight = -0.1  # 脚が接地しているのに滑るとペナルティが与えられる
        self.undesired_contacts.weight = -1.0  # 頭や太ももなどが地面に接触するとペナルティが与えられる
        self.stand_still.weight = -1.0  # 停止中に標準姿勢から崩れていたらペナルティが与えられる


@configclass
class Go2ParkourClimbUpEnvCfg(LocomotionVelocityRoughEnvCfg):
    num_envs: int = 4096
    env_spacing: float = 2.5
    scene: Go2ParkourClimbUpSceneCfg = Go2ParkourClimbUpSceneCfg()
    observations: Go2ParkourObservationsCfg = Go2ParkourObservationsCfg()
    rewards: Go2ParkourClimbUpRewardsCfg = Go2ParkourClimbUpRewardsCfg()

    def __post_init__(self):
        self.scene.num_envs = self.num_envs
        self.scene.env_spacing = self.env_spacing

        super().__post_init__()

        # 「タイル数ぶんの壁」を生成
        # virtual_terrain.make_virtual_climb_track_cfg(
        #     terrain_cfg=self.scene.terrain.terrain_generator,
        #     wall_offset_x=0.0,   # 各タイル内で壁を置きたい x [m]
        #     wall_offset_y=0.0,
        #     length=4.0,
        #     height=1.0,
        #     thickness=0.1,
        # )


@configclass
class Go2ParkourClimbUpEnvCfg_PLAY(Go2ParkourClimbUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
