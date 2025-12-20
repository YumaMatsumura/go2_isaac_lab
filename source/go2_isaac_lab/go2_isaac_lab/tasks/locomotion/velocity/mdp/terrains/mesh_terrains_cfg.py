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


@configclass
class ScatteredPipesTerrainCfg(SubTerrainBaseCfg):
    """円形パイプ散乱地形の設定"""

    function = mesh_terrains.mesh_scattered_pipes

    # 再現性
    seed: int = 0

    # 地面の厚み（見た目/衝突用）
    ground_thickness: float = 1.0

    # パイプ本数（difficultyで線形補間）
    num_pipes_range: tuple[int, int] = (10, 80)

    # パイプ半径（m）
    pipe_radius_range: tuple[float, float] = (0.03, 0.08)

    # パイプ長さ（m）
    pipe_length_range: tuple[float, float] = (0.25, 0.9)

    # 置くときの端マージン（m）
    border_margin: float = 0.25

    # 中央に安全地帯を作る（半径m）。0で無効
    safe_center_radius: float = 0.6

    # 近接防止（パイプ中心間のざっくりクリアランス）
    pipe_clearance: float = 0.05

    # 置けないときのリトライ上限
    max_placement_tries: int = 200

    # 円柱の分割数（見た目/メッシュ密度）
    pipe_sections: int = 24

    # 横倒し後の傾きの最大（rad）。difficultyでスケール
    max_tilt_rad: float = 0.35  # 約20度

    # 地面へ少し埋める量（m）
    embed_depth: float = 0.02
