import isaaclab.terrains as terrain_gen
from go2_isaac_lab.assets.go2 import GO2_CFG
from go2_isaac_lab.tasks.locomotion.velocity import mdp
from go2_isaac_lab.tasks.locomotion.velocity.go2_isaac_lab_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
)
from isaaclab.utils import configclass


@configclass
class Go2SceneCfg(MySceneCfg):

    def __post_init__(self):
        super().__post_init__()

        # set the robot as mevius
        self.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain parameter settings
        self.terrain.max_init_terrain_level = 5
        self.terrain.terrain_generator.size = (8.0, 8.0)
        self.terrain.terrain_generator.border_width = 20.0
        self.terrain.terrain_generator.num_rows = 10  # 難易度の数
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.1
        self.terrain.terrain_generator.vertical_scale = 0.005
        self.terrain.terrain_generator.slope_threshold = 0.75
        self.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.terrain.terrain_generator.curriculum = True
        self.terrain.terrain_generator.use_cache = False
        self.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.1
        self.terrain.terrain_generator.sub_terrains["random_rough"] = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        )
        self.terrain.terrain_generator.sub_terrains["climb_down"] = mdp.ClimbDownTerrainCfg(
            proportion=0.80,
            box_height_range=(0.05, 0.70),
            box_edge=1.5,
        )


@configclass
class Go2ParkourEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2ParkourEnvCfg_PLAY(Go2ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
