import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


def _terrain_tile_center_to_world_xy(
    row: int,
    col: int,
    tile_size_x: float,
    tile_size_y: float,
    num_rows: int,
    num_cols: int,
) -> tuple[float, float]:
    total_x = num_rows * tile_size_x
    total_y = num_cols * tile_size_y

    center_x = (row + 0.5) * tile_size_x - 0.5 * total_x
    center_y = (col + 0.5) * tile_size_y - 0.5 * total_y

    return center_x, center_y


def make_virtual_climb_track_cfg(
    terrain_cfg: TerrainGeneratorCfg,
    wall_offset_x: float = 2.0,
    wall_offset_y: float = 0.0,
    length: float = 4.0,
    height: float = 0.4,
    thickness: float = 0.1,
):
    tile_size_x = float(terrain_cfg.size[0])
    tile_size_y = float(terrain_cfg.size[1])

    num_rows = int(terrain_cfg.num_rows)
    num_cols = int(terrain_cfg.num_cols)

    # total_x = num_rows * tile_size_x
    # total_y = num_cols * tile_size_y

    box_cfg = sim_utils.CuboidCfg(
        size=(thickness, length, height),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0),
        ),
    )

    for row in range(num_rows):
        for col in range(num_cols):
            center_x, center_y = _terrain_tile_center_to_world_xy(
                row=row,
                col=col,
                tile_size_x=tile_size_x,
                tile_size_y=tile_size_y,
                num_rows=num_rows,
                num_cols=num_cols,
            )

            x = center_x + wall_offset_x
            y = center_y + wall_offset_y
            z = 0.5 * height  # 地面から高さの半分だけ持ち上げる

            prim_path = f"/World/VirtualClimbTrack/row_{row}_col_{col}"
            box_cfg.func(
                prim_path,
                box_cfg,
                translation=(x, y, z),
            )
