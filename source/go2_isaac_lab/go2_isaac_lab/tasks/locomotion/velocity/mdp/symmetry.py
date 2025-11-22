# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for Unitree Go2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    two symmetrical transformations: original and left-right. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the Unitree Go2 robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    batch_size, total_dim = obs.shape

    BASE_BLOCK_DIM = 45

    if total_dim % BASE_BLOCK_DIM != 0:
        raise RuntimeError(
            f"[symmetry] Unexpected policy obs dim {total_dim}. "
            f"Expected a multiple of {BASE_BLOCK_DIM} "
            "(base + EMA blocks)."
        )

    num_blocks = total_dim // BASE_BLOCK_DIM

    def _transform_block(block: torch.Tensor) -> torch.Tensor:
        """1つの45次元ブロックに対して左右対称変換を適用する"""
        b = block
        # ang vel
        b[:, :3] = b[:, :3] * torch.tensor([-1, 1, -1], device=device)
        # projected gravity
        b[:, 3:6] = b[:, 3:6] * torch.tensor([1, -1, 1], device=device)
        # velocity command
        b[:, 6:9] = b[:, 6:9] * torch.tensor([1, -1, -1], device=device)
        # joint pos
        b[:, 9:21] = _switch_go2_joints_left_right(b[:, 9:21])
        # joint vel
        b[:, 21:33] = _switch_go2_joints_left_right(b[:, 21:33])
        # last actions
        b[:, 33:45] = _switch_go2_joints_left_right(b[:, 33:45])
        return b

    for i in range(num_blocks):
        start = i * BASE_BLOCK_DIM
        end = start + BASE_BLOCK_DIM
        obs[:, start:end] = _transform_block(obs[:, start:end])

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    # if "height_scan" in env.observation_manager.active_terms["policy"]:
    #     height_scan = obs[:, 45:232]
    #     if height_scan.numel() == obs.shape[0] * (11 * 17):
    #         obs[:, 45:232] = height_scan.view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    Unitree Go2 robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_go2_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
]

Correspondingly, the joint ordering for the Unitree Go2 robot is:

* FR = front right --> [0, 1, 2]
* FL = front left --> [3, 4, 5]
* RR = rear right --> [6, 7, 8]
* RL = rear left --> [9, 10, 11]
"""


def _switch_go2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # right <-- left
    joint_data_switched[..., [0, 1, 2, 6, 7, 8]] = joint_data[..., [3, 4, 5, 9, 10, 11]]
    # left <-- right
    joint_data_switched[..., [3, 4, 5, 9, 10, 11]] = joint_data[..., [0, 1, 2, 6, 7, 8]]

    # Flip the sign of the hip joints
    joint_data_switched[..., [0, 3, 6, 9]] *= -1.0

    return joint_data_switched
