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


@configclass
class SquarePitPowerTerrainCfg(HfTerrainBaseCfg):
    """
    “中心へ行くほど低くなる”正方形ピット（四角い穴）の設定。
    - function              : height-field 生成関数（hf_terrains.hf_exp_square_pit）
    - pit_edge              : 穴の一辺 [m]（内部のフラット/低地の代表幅）
    - rim_height_range      : 外周（リム）の高さレンジ [m]（difficulty 0→1 で H_min→H_max）
    - rise_length_range        : 指数減衰の e 折れ長さレンジ [m]（0→1 で L_max→L_min; 小さいほど急）
    - growth_exponent       : 立ち上がりのべき指数 p（>1で凸、境界から離れるほど傾斜角度が増加）
    - height_cutoff         : 極小高さカット [m]（これ未満は 0 とみなす）
    - noise_std             : 高さノイズの標準偏差 [m]（荒れ地感、0 で無効）

    親クラス(HfTerrainBaseCfg)が持つ主な共通パラメータ:
    - size                  : サブテレインの (X, Y) サイズ [m]
    - horizontal_scale      : ピクセル解像度（1px の物理長さ）[m/px]
    - vertical_scale        : 高さ量子化（1 step の物理長さ）[m/step]
    - border_height, border_size なども必要に応じて利用可
    """

    function = hf_terrains.hf_square_pit_power

    # --- 形状パラメータ ---
    pit_edge: float = 1.0

    # --- カリキュラムで変化させるレンジ ---
    rim_height_range: tuple[float, float] = (0.2, 1.0)
    rise_length_range: tuple[float, float] = (1.0, 0.02)

    # --- 形状コントロール ---
    growth_exponent: float = 2.0

    # --- 実用パラメータ ---
    height_cutoff: float = 1e-4
    noise_std: float = 0.0
