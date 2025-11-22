from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.ray_caster import RayCaster


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


# 参考: https://github.com/yang-zj1026/legged-loco/blob/904bbfd9d3545e5d11d4dfdb98db1dfa355f10ac/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/leggedloco/mdp/observations.py#L281-L384
# -----
# LiDAR高さマップ用のパラメータ
# -----
# x-y平面のボクセルサイズ [m]
voxel_size_xy = 0.06

# LiDARローカル座標系で見たときのx, y, zの有効範囲 [m]
range_x = [-0.8, 0.2 + 1e-9]
range_y = [-0.8, 0.8 + 1e-9]
range_z = [0.0, 5.0]

from collections import deque

# Create a deque with a maximum length of 10
prev_height_maps = deque(maxlen=10)


def height_map_lidar(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # sensorを取得
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # world座標でのヒット位置 - センサー位置 = センサー原点から見たベクトル
    hit_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    hit_vec[torch.isinf(hit_vec)] = 0.0
    hit_vec[torch.isnan(hit_vec)] = 0.0

    hit_vec_shape = hit_vec.shape
    hit_vec = hit_vec.view(-1, hit_vec.shape[-1])

    # ロボットベース姿勢（world）
    robot_base_quat_w = env.scene["robot"].data.root_quat_w

    # ロボットベース -> LiDARの固定オフセット（クォータニオン）
    # TODO(YumaMatsumura): 固定値になっている
    sensor_quat_default = (
        torch.tensor([-0.131, 0.0, -0.991, 0.0], device=robot_base_quat_w.device)
        .unsqueeze(0)
        .repeat(hit_vec_shape[0], 1)
    )

    # worldから見たLiDARの向き
    sensor_quat_w = math_utils.quat_mul(robot_base_quat_w, sensor_quat_default)

    # 各レイに対応するようにクォータニオンを複製
    quat_w_dup = (sensor_quat_w.unsqueeze(1).repeat(1, hit_vec_shape[1], 1)).view(-1, sensor_quat_w.shape[-1])

    # worldベクトル -> LiDARローカル座標系に変換
    hit_vec_lidar_frame = math_utils.quat_rotate_inverse(quat_w_dup, hit_vec)
    hit_vec_lidar_frame = hit_vec_lidar_frame.view(hit_vec_shape[0], hit_vec_shape[1], hit_vec_lidar_frame.shape[-1])

    num_envs = hit_vec_lidar_frame.shape[0]

    # x, yのビン（グリッド境界）を作成
    x_bins = torch.arange(range_x[0], range_x[1], voxel_size_xy, device=hit_vec_lidar_frame.device)
    y_bins = torch.arange(range_y[0], range_y[1], voxel_size_xy, device=hit_vec_lidar_frame.device)

    x = hit_vec_lidar_frame[..., 0]
    y = hit_vec_lidar_frame[..., 1]
    z = hit_vec_lidar_frame[..., 2]

    # 有効範囲内の点だけを使用
    valid_indices = (
        (x > range_x[0])
        & (x <= range_x[1])
        & (y > range_y[0])
        & (y <= range_y[1])
        & (z >= range_z[0])
        & (z <= range_z[1])
    )

    x_filtered = x[valid_indices]
    y_filtered = y[valid_indices]
    z_filtered = z[valid_indices]

    # x, yがどのビンに入るかを計算
    x_indices = torch.bucketize(x_filtered, x_bins) - 1
    y_indices = torch.bucketize(y_filtered, y_bins) - 1

    # 環境インデックス（0..num_envs-1）を展開
    env_indices = torch.arange(num_envs, device=hit_vec_lidar_frame.device).unsqueeze(1).expand_as(valid_indices)
    flat_env_indices = env_indices[valid_indices]

    # 2.5Dマップをinfで初期化
    map_2_5D = torch.full((num_envs, len(x_bins), len(y_bins)), float("inf"), device=hit_vec_lidar_frame.device)

    # (env, x, y) -> 1次元インデックス
    linear_indices = flat_env_indices * len(x_bins) * len(y_bins) + x_indices * len(y_bins) + y_indices

    # 各セルに対して最小z（最も低い高さ＝地面/段差表面）を格納し、オフセットを引く
    map_2_5D = map_2_5D.view(-1).scatter_reduce_(0, linear_indices, z_filtered, reduce="amin") - offset

    # 小さい値を0に丸めてノイズを削減
    map_2_5D = torch.where(map_2_5D < 0.05, torch.tensor(0.0, device=map_2_5D.device), map_2_5D)

    # データがなかったセル（inf）は0にする
    map_2_5D = torch.where(torch.isinf(map_2_5D), torch.tensor(0.0, device=map_2_5D.device), map_2_5D)

    # (num_envs, H, W)に戻す
    map_2_5D = map_2_5D.view(num_envs, len(x_bins), len(y_bins))

    # 3x3の近傍で最大値をとる（障害物の膨張・穴埋めのような効果）
    max_across_frames = F.max_pool2d(map_2_5D, kernel_size=3, stride=1, padding=1).view(num_envs, -1)

    return max_across_frames


# ----- EMA -----
def _init_ema_single_buffer(
    env: ManagerBasedRLEnv,
    key: str,
    obs: torch.Tensor,
) -> None:
    """1つの観測用 EMA バッファを env に用意。"""
    if not hasattr(env, "_ema_single_buffers"):
        env._ema_single_buffers: dict[str, torch.Tensor] = {}

    if key not in env._ema_single_buffers:
        env._ema_single_buffers[key] = torch.zeros_like(obs, device=obs.device)


def _update_ema(prev_ema: torch.Tensor, current: torch.Tensor, alpha: float) -> torch.Tensor:
    """ema_t = alpha * ema_{t-1} + (1 - alpha) * x_t"""
    return alpha * prev_ema + (1.0 - alpha) * current


def ema_single(
    env: ManagerBasedRLEnv,
    base_func,
    key: str,
    alpha: float = 0.5,
    base_params: dict[str, Any] | None = None,
) -> torch.Tensor:
    """任意の観測関数 base_func(env, **base_params) に対して EMA を1個だけ返す。

    - base_func: 既存の観測関数（例: mdp.joint_pos_rel）
    - key: この観測用の識別子（例: "joint_pos_rel"）
    """
    if base_params is None:
        base_params = {}

    # もともとの観測をそのまま計算
    base_obs = base_func(env, **base_params)  # shape: [num_envs, obs_dim]

    # バッファ初期化（最初の1回だけ）
    _init_ema_single_buffer(env, key, base_obs)

    prev_ema = env._ema_single_buffers[key]
    new_ema = _update_ema(prev_ema, base_obs, float(alpha))
    env._ema_single_buffers[key] = new_ema

    return new_ema


# ----- ここからは、特権的観測用のobservations -----
def friction_coeff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*"),
    mode: str = "mean",  # "static", "dynamic", "mean"
) -> torch.Tensor:
    """ランダム化済みの摩擦係数を1次元の観測として返す"""

    asset: Articulation = env.scene[asset_cfg.name]

    mat_props = asset.root_physx_view.get_material_properties().to(env.device)

    static_friction = mat_props[..., 0]
    dynamic_friction = mat_props[..., 1]

    if mode == "static":
        friction = static_friction
    elif mode == "dynamic":
        friction = dynamic_friction
    else:
        friction = 0.5 * (static_friction + dynamic_friction)

    # body_namesの指定があれば、そのリンクだけに絞る
    if asset_cfg.body_ids is not None and len(asset_cfg.body_ids) > 0:
        friction = friction[:, asset_cfg.body_ids]

    # 全身の平均値を1要素として返す
    friction_mean = torch.mean(friction, dim=1, keepdim=True)
    return friction_mean


def body_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base"]),
) -> torch.Tensor:
    """指定したbody_namesの質量の合計値を返す"""

    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)

    # body_idsに絞って合計
    selected = masses[:, asset_cfg.body_ids]
    mass_sum = torch.sum(selected, dim=1, keepdim=True)
    return mass_sum


def payload_pos_rel_to_base(
    env: ManagerBasedRLEnv,
    payload_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["payload"]),
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base"]),
) -> torch.Tensor:
    """baseから見たpayloadの相対位置[x, y, z]を返す"""

    asset: Articulation = env.scene[payload_cfg.name]

    body_pos = asset.data.body_pos_w.to(env.device)

    payload_pos = body_pos[:, payload_cfg.body_ids[0], :]
    base_pos = body_pos[:, base_cfg.body_ids[0], :]

    rel_pos = payload_pos - base_pos
    return rel_pos
