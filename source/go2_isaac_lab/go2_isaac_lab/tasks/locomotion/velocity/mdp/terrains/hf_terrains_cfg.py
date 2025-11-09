from dataclasses import MISSING

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains


@configclass
class ClimbDownTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.hf_climb_down
    box_height_range: tuple[float, float] = MISSING
    edge_offset: float = MISSING  # 台の端までの長さ [m]
