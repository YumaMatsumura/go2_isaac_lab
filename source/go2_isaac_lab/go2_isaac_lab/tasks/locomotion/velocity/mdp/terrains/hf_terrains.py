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
    edge = int(round(cfg.edge_offset / cfg.horizontal_scale))
    edge = int(np.clip(edge, 0, width_pixels))

    # 手前に台、奥は地面
    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    height_field[:edge, :] = np.int16(height_steps)
    return height_field
