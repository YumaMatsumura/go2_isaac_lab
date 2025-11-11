from dataclasses import MISSING

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains


@configclass
class ClimbDownTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.hf_climb_down
    box_height_range: tuple[float, float] = MISSING  # 段差高さの範囲 [m]
    box_edge: float = MISSING  # 台の1辺の長さ [m]


@configclass
class ClimbUpTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.hf_climb_down
    box_height_range: tuple[float, float] = MISSING  # 段差高さの範囲 [m]
    box_edge: float = MISSING  # 台の1辺の長さ [m]
