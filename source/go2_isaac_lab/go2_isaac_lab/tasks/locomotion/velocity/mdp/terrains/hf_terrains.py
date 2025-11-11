from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def hf_climb_down(difficulty: float, cfg: hf_terrains_cfg.ClimbDownTerrainCfg) -> np.ndarray:
    min_height, max_height = cfg.box_height_range
    box_height = min_height + difficulty * (max_height - min_height)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_steps = int(round(box_height / cfg.vertical_scale))

    box_edge_pixels = int(round(cfg.box_edge / cfg.horizontal_scale))
    box_edge_pixels = int(np.clip(box_edge_pixels, 1, min(width_pixels, length_pixels)))

    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    cx = width_pixels // 2
    cy = length_pixels // 2
    half = box_edge_pixels // 2
    x0_f, x1_f = np.clip([cx - half, cx - half + box_edge_pixels], 0, width_pixels)
    y0_f, y1_f = np.clip([cy - half, cy - half + box_edge_pixels], 0, length_pixels)
    x0, x1 = int(x0_f), int(x1_f)
    y0, y1 = int(y0_f), int(y1_f)

    height_field[x0:x1, y0:y1] = np.int16(height_steps)
    return height_field


@height_field_to_mesh
def hf_climb_up(difficulty: float, cfg: hf_terrains_cfg.ClimbUpTerrainCfg) -> np.ndarray:
    min_height, max_height = cfg.box_height_range
    box_height = min_height + difficulty * (max_height - min_height)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_steps = int(round(box_height / cfg.vertical_scale))

    box_edge_pixels = int(round(cfg.box_edge / cfg.horizontal_scale))
    box_edge_pixels = int(np.clip(box_edge_pixels, 1, min(width_pixels, length_pixels)))

    height_field = np.full((width_pixels, length_pixels), np.int16(height_steps), dtype=np.int16)

    cx = width_pixels // 2
    cy = length_pixels // 2
    half = box_edge_pixels // 2
    x0_f, x1_f = np.clip([cx - half, cx - half + box_edge_pixels], 0, width_pixels)
    y0_f, y1_f = np.clip([cy - half, cy - half + box_edge_pixels], 0, length_pixels)
    x0, x1 = int(x0_f), int(x1_f)
    y0, y1 = int(y0_f), int(y1_f)

    height_field[x0:x1, y0:y1] = np.int16(0)
    return height_field
