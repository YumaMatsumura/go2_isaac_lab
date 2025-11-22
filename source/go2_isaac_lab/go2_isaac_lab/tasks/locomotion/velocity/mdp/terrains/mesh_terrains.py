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
