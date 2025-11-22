from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import mesh_terrains


@configclass
class CrouchTerrainCfg(SubTerrainBaseCfg):
    """Crouch Terrainの設定
    Parameters
    ----------
    inner_width : float
        - 天井がない中央の広場の幅の半分
    ceil_width: float
        - 天井リングの幅（上下左右方向の幅）
    ceil_height_range: tup;e[float, float]
        - 天井高さのレンジ（(最小高さ, 最大高さ)で与える）
    """

    function = mesh_terrains.mesh_crouch

    inner_width: float = 0.6
    ceil_width: float = 1.0
    ceil_height_range: tuple[float, float] = (0.7, 1.0)
