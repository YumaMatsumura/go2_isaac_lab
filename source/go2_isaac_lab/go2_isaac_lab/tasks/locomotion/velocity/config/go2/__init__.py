# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, parkour_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Go2-Isaac-Lab-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Go2RoughEnvCfg,
        "play_env_cfg_entry_point": rough_env_cfg.Go2RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Isaac-Lab-Velocity-Parkour-Stationary-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.Go2ParkourStationaryEnvCfg,
        "play_env_cfg_entry_point": parkour_env_cfg.Go2ParkourStationaryEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourPPORunnerWithSymmetryCfg",
    },
)

gym.register(
    id="Go2-Isaac-Lab-Velocity-Parkour-Crouch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.Go2ParkourCrouchEnvCfg,
        "play_env_cfg_entry_point": parkour_env_cfg.Go2ParkourCrouchEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourPPORunnerWithSymmetryCfg",
    },
)

gym.register(
    id="Go2-Isaac-Lab-Velocity-Parkour-ClimbUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": parkour_env_cfg.Go2ParkourClimbUpEnvCfg,
        "play_env_cfg_entry_point": parkour_env_cfg.Go2ParkourClimbUpEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourPPORunnerWithSymmetryCfg",
    },
)
