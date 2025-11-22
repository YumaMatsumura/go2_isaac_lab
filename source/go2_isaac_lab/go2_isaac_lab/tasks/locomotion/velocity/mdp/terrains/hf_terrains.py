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


@height_field_to_mesh
def hf_square_pit_power(difficulty: float, cfg: hf_terrains_cfg.SquarePitPowerTerrainCfg) -> np.ndarray:
    """
    中央へ行くほど低くなる“正方形の穴”地形を生成する。
    - H: リム高さ（difficulty で内挿）
    - L: 立ち上がりスケール（difficulty で内挿、L が小さいほど急峻）
    - p: 立ち上がりのべき指数（p>1 で凸：境界から離れるほど傾斜が強くなる）
    """

    # -----------------------------
    # 基本スケールと解像度
    # -----------------------------
    hscale = float(cfg.horizontal_scale)  # [m / pixel]
    vscale = float(cfg.vertical_scale)  # [m / step]

    width_px = int(round(cfg.size[0] / hscale))  # X 方向ピクセル数
    length_px = int(round(cfg.size[1] / hscale))  # Y 方向ピクセル数

    # -----------------------------
    # difficulty によるパラメータ内挿
    # -----------------------------
    H_min, H_max = cfg.rim_height_range
    H = float(H_min + difficulty * (H_max - H_min))  # リム高さ [m]

    L_max, L_min = cfg.rise_length_range  # e 折れ長さ [m]
    L = float(L_max + difficulty * (L_min - L_max))  # 小さいほど急峻
    L = max(L, 1e-6)  # 数値安定化

    # -----------------------------
    # 正方形“穴”のサイズ（ピクセル）
    # -----------------------------
    edge_px = int(round(cfg.pit_edge / hscale))  # 一辺のピクセル数
    edge_px = int(np.clip(edge_px, 1, min(width_px, length_px)))  # 範囲クリップ
    half_edge_px = edge_px // 2

    # -----------------------------
    # 出力バッファ（height field: int16 steps）
    # -----------------------------
    height_field = np.zeros((width_px, length_px), dtype=np.int16)

    # 画素中心座標での原点を画像中心に取る
    cx = width_px * 0.5
    cy = length_px * 0.5

    # -----------------------------
    # 高さの計算ループ
    # -----------------------------
    p = cfg.growth_exponent
    for ix in range(width_px):
        dx = abs((ix + 0.5) - cx)  # 中心からの X 距離 [px]
        for iy in range(length_px):
            dy = abs((iy + 0.5) - cy)  # 中心からの Y 距離 [px]

            # 正方形境界からの距離（px）
            #   s_px > 0  : 正方形の外（リム側）
            #   s_px <= 0 : 正方形の内（穴側, 底の一辺 = half_edge_px）
            l_inf = max(dx, dy)
            s_px = l_inf - half_edge_px

            if s_px <= 0:
                # 穴の内側：底（高さ0）
                height_m = 0
            else:
                # リム側：境界から離れるほど強い（凸）立ち上がり
                t = (s_px * hscale) / L
                height_m = H * (t**p)
                if height_m > H:
                    height_m = H

            # 微小値のカットオフ（ノイズ抑制・高速化）
            if height_m < cfg.height_cutoff:
                height_m = 0.0

            # 荒れ地感の微小ノイズ（任意）
            if cfg.noise_std > 0.0 and height_m > 0.0:
                height_m = max(0.0, height_m + np.random.normal(0.0, cfg.noise_std))

            # [m] -> [steps] へ量子化し、int16 にクリップ
            steps = int(np.round(height_m / vscale))
            steps = int(np.clip(steps, np.iinfo(np.int16).min, np.iinfo(np.int16).max))
            height_field[ix, iy] = np.int16(steps)

    return height_field
