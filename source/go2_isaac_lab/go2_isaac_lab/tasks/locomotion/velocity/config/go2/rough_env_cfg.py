import isaaclab.terrains as terrain_gen
from go2_isaac_lab.assets.go2 import GO2_CFG
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
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.1
        self.terrain.terrain_generator.vertical_scale = 0.005
        self.terrain.terrain_generator.slope_threshold = 0.75
        self.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.terrain.terrain_generator.use_cache = False
        self.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.1
        self.terrain.terrain_generator.sub_terrains["random_rough"] = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        )
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"] = terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        )
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"] = (
            terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            )
        )
        self.terrain.terrain_generator.sub_terrains["boxes"] = terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        )
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"] = terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        )
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"] = (
            terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            )
        )


@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2RoughEnvCfg_PLAY(Go2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
