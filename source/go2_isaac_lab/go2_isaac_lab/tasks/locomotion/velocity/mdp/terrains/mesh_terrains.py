from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def mesh_crouch(difficulty: float, cfg: mesh_terrains_cfg.CrouchTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """中央に天井がない正方形の通路があり、その周囲を天井で囲う地形。
    difficultyが高いほど天井が低くなり、しゃがみ動作を要求する。

      # ------------------------ #
      |                          |
      |                          |
      |   # ---------------- #   |
      |   |      inner_width | ceil_width
      |   |         <------->|<->|
      |   |                  |   |
      |   # ---------------- #   |
      |                          |
      |                          |
      # ------------------------ #

    """

    meshes_list: list[trimesh.Trimesh] = []

    # 1) 共通パラメータ
    size_x, size_y = cfg.size
    center_x = 0.5 * size_x
    center_y = 0.5 * size_y

    # 2) difficultyから天井高さを決める
    min_ceil_height, max_ceil_height = cfg.ceil_height_range
    ceil_height = max_ceil_height - difficulty * (max_ceil_height - min_ceil_height)

    # 3) 地面を作る
    terrain_height = 1.0
    ground_dim = (size_x, size_y, terrain_height)
    ground_pos = (center_x, center_y, -terrain_height * 0.5)
    ground = trimesh.creation.box(ground_dim, trimesh.transformations.translation_matrix(ground_pos))
    meshes_list.append(ground)

    # 4) 天井を作る
    inner_width = cfg.inner_width
    ceil_width = cfg.ceil_width
    ceil_thickness = 0.2
    center_z = ceil_height + ceil_thickness * 0.5

    top_y = center_y + inner_width + ceil_width * 0.5
    bottom_y = center_y - inner_width - ceil_width * 0.5
    left_x = center_x - inner_width - ceil_width * 0.5
    right_x = center_x + inner_width + ceil_width * 0.5

    def add_box(dim_x: float, dim_y: float, pos_x: float, pos_y: float) -> None:
        dim = (dim_x, dim_y, ceil_thickness)
        pos = (pos_x, pos_y, center_z)
        mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(mesh)

    # 左上
    add_box(ceil_width, ceil_width, left_x, top_y)
    # 右上
    add_box(ceil_width, ceil_width, right_x, top_y)
    # 左下
    add_box(ceil_width, ceil_width, left_x, bottom_y)
    # 右下
    add_box(ceil_width, ceil_width, right_x, bottom_y)
    # 上
    add_box(inner_width * 2.0, ceil_width, center_x, top_y)
    # 下
    add_box(inner_width * 2.0, ceil_width, center_x, bottom_y)
    # 左
    add_box(ceil_width, inner_width * 2.0, left_x, center_y)
    # 右
    add_box(ceil_width, inner_width * 2.0, right_x, center_y)

    origin = np.array([center_x, center_y, 0.0])

    return meshes_list, origin


def _rand_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return trimesh.transformations.rotation_matrix(angle, axis)


def mesh_scattered_pipes(
    difficulty: float,
    cfg: mesh_terrains_cfg.ScatteredPipesTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """円形パイプ（円柱）がランダムに散乱している不整地を生成する。

    - 地面: box
    - パイプ: cylinder を横倒しにしてランダム配置
    - difficulty が高いほど、パイプ本数・太さ・高さ(=直径)や起伏感が増える想定

    Returns
    -------
    meshes_list : list[trimesh.Trimesh]
        地形メッシュのリスト
    origin : np.ndarray
        地形の基準点（中心）
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    rng = np.random.default_rng(cfg.seed)

    meshes_list: list[trimesh.Trimesh] = []

    # --- 共通 ---
    size_x, size_y = cfg.size
    center_x = 0.5 * size_x
    center_y = 0.5 * size_y

    # --- 地面 ---
    terrain_height = cfg.ground_thickness
    ground_dim = (size_x, size_y, terrain_height)
    ground_pos = (center_x, center_y, -terrain_height * 0.5)
    ground = trimesh.creation.box(ground_dim, trimesh.transformations.translation_matrix(ground_pos))
    meshes_list.append(ground)

    # --- difficulty からパイプのパラメータを作る ---
    # 本数
    n_min, n_max = cfg.num_pipes_range
    num_pipes = int(round(n_min + difficulty * (n_max - n_min)))

    # 半径（m）
    r_min, r_max = cfg.pipe_radius_range
    # 長さ（m）
    l_min, l_max = cfg.pipe_length_range

    # 横倒し度合い：横倒し前提 + 小さくロール/ピッチを乱す
    max_tilt = cfg.max_tilt_rad * difficulty  # 難しいほど傾きが増える

    # 配置可能領域（壁ぎわに置かない・中心安全地帯など）
    margin = cfg.border_margin
    safe_radius = cfg.safe_center_radius  # 中央に置かないための半径（0で無効）
    x_lo, x_hi = margin, size_x - margin
    y_lo, y_hi = margin, size_y - margin

    # 近接しすぎ防止（簡易）：中心距離が近いものを弾く
    placed_xy: list[tuple[float, float, float]] = []  # (x, y, approx_radius)
    max_tries = max(cfg.max_placement_tries, num_pipes * 20)

    def sample_xy(approx_clearance: float) -> tuple[float, float] | None:
        for _ in range(max_tries):
            x = _rand_uniform(rng, x_lo, x_hi)
            y = _rand_uniform(rng, y_lo, y_hi)

            # 中央セーフゾーン
            if safe_radius > 0.0:
                dx = x - center_x
                dy = y - center_y
                if (dx * dx + dy * dy) < (safe_radius * safe_radius):
                    continue

            ok = True
            for px, py, pr in placed_xy:
                ddx = x - px
                ddy = y - py
                # ざっくり：半径の和 + クリアランス
                if (ddx * ddx + ddy * ddy) < (pr + approx_clearance) ** 2:
                    ok = False
                    break
            if ok:
                return x, y
        return None

    # --- パイプ生成 ---
    for _ in range(num_pipes):
        radius = _rand_uniform(rng, r_min, r_max)
        length = _rand_uniform(rng, l_min, l_max)

        # 位置（近接防止は半径スケールで適当に）
        approx_clearance = cfg.pipe_clearance + radius * 2.0
        xy = sample_xy(approx_clearance=approx_clearance)
        if xy is None:
            # 置けなかったら諦める（地形生成が止まるのが一番つらいので）
            continue
        x, y = xy
        placed_xy.append((x, y, approx_clearance))

        # Cylinder は +Z 方向に軸を持つ（trimeshの作り方依存）ので、
        # パイプらしくするために基本は X か Y 軸方向へ 90deg 回転させて横倒しにする
        axis_choice = rng.integers(0, 2)  # 0: X向き, 1: Y向き
        if axis_choice == 0:
            # Z軸→X軸へ（Y軸回りに +90deg）
            base_rot = _rotation_matrix_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 2.0)
        else:
            # Z軸→Y軸へ（X軸回りに -90deg）
            base_rot = _rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), -np.pi / 2.0)

        # さらに、yaw（水平回転）と微小 tilt を加えて自然に
        yaw = _rand_uniform(rng, 0.0, 2.0 * np.pi)
        yaw_rot = _rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)

        # tilt：横倒しのパイプが少し浮いたり、片側が乗り上がったりする感じ
        tilt_axis = rng.normal(size=3)
        tilt_axis[2] = 0.0  # 水平面内の軸で傾けると「片側が上がる」感じになりやすい
        if np.linalg.norm(tilt_axis) < 1e-6:
            tilt_axis = np.array([1.0, 0.0, 0.0])
        tilt_angle = _rand_uniform(rng, -max_tilt, max_tilt)
        tilt_rot = _rotation_matrix_from_axis_angle(tilt_axis, tilt_angle)

        # メッシュ本体
        pipe_mesh = trimesh.creation.cylinder(
            radius=radius,
            height=length,
            sections=cfg.pipe_sections,
        )

        # Z位置：地面に少し埋める/少し浮かす（物理的に安定しやすい）
        # 横倒し後の「最低点」が地面に近い位置に来るのが理想なので、ここでは簡易に
        # 「中心が半径分だけ上」に置いてから、difficultyに応じて埋め込み量を変える
        embed = cfg.embed_depth * (0.3 + 0.7 * difficulty)  # 難しいほど引っかかりやすい
        z = radius - embed

        # 最終変換：T * yaw * tilt * base
        T = trimesh.transformations.translation_matrix([x, y, z])
        M = trimesh.transformations.concatenate_matrices(T, yaw_rot, tilt_rot, base_rot)
        pipe_mesh.apply_transform(M)

        meshes_list.append(pipe_mesh)

    origin = np.array([center_x, center_y, 0.0], dtype=np.float32)
    return meshes_list, origin
