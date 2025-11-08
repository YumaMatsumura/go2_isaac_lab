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
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.terrain.terrain_generator.sub_terrains["random_rough"].border_width = 0.25
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.4)
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].platform_width = 2.0
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].border_width = 0.25
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.4)
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].platform_width = 2.0
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].border_width = 0.25
        self.terrain.terrain_generator.sub_terrains["boxes"].grid_width = 0.45
        self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.20)
        self.terrain.terrain_generator.sub_terrains["boxes"].platform_width = 2.0
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.23)
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_width = 0.3
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].platform_width = 3.0
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].border_width = 1.0
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].holes = False
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.23)
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_width = 0.3
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].platform_width = 3.0
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].border_width = 1.0
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].holes = False

        # set the terrain proportions
        self.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.1
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.1
        self.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1
        self.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.2
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.2
        self.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.2


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
