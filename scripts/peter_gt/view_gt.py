"""Viser viewer for the climb_wall_3 ground-truth scenes.

Shows one set at a time: the metric coloured point cloud from ``scene.npz`` or, with
the ``cloud`` dropdown, the cleaned ``scene_clean.npz`` (the three VGGT views, toggled
separately), the VGGT camera frustums in orange and —
where ``calibration.npz`` exists — the chessboard-calibrated ones in cyan, so the
two solutions can be eyeballed against each other. Each frustum is drawn from that
camera's full K, so a shifted principal point or fx != fy is visible as a leaning
or non-square pyramid. World = cam_left, metres.

    uv run --no-sync python scripts/peter_gt/view_gt.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vt

GT_ROOT = Path(__file__).resolve().parents[2] / "peter/climb_wall_3_gt/scene"
VIEW_NAMES = ["cam_left", "cam_right", "cam_3"]
VGGT_COLOR = (255, 140, 0)
CALIB_COLOR = (0, 200, 255)
FRUSTUM_SCALE = 0.15


def camera_pose(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-from-world OpenCV ``[R|t]`` -> (world position, world-from-cam wxyz)."""
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return -R.T @ t, vt.SO3.from_matrix(R.T).wxyz


def frustum_segments(extr: np.ndarray, K: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Line segments (8, 2, 3) of the camera's viewing pyramid, drawn from the full K.

    The corners are the rays through the four image corners, so an off-centre
    principal point makes the pyramid lean and fx != fy makes it non-square —
    both invisible if the frustum were built from one field of view and an aspect.
    """
    w, h = size
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    rays = (np.linalg.inv(K) @ np.c_[corners, np.ones(4)].T).T * FRUSTUM_SCALE
    R, t = extr[:3, :3], extr[:3, 3]
    world = (rays - t) @ R                              # camera -> world
    apex = -R.T @ t
    segs = [[apex, world[i]] for i in range(4)]
    segs += [[world[i], world[(i + 1) % 4]] for i in range(4)]
    return np.array(segs)


def add_frustums(server: viser.ViserServer, path: str, extr: np.ndarray, K: np.ndarray,
                 size: np.ndarray, color: tuple[int, int, int]) -> list:
    handles = []
    for k, name in enumerate(VIEW_NAMES):
        segs = frustum_segments(extr[k].astype(np.float64), K[k].astype(np.float64),
                                (int(size[k][0]), int(size[k][1])))
        handles.append(server.scene.add_line_segments(
            f"{path}/{name}", points=segs, colors=color, line_width=2.0))
    return handles


class Viewer:
    """One scene at a time; switching the set tears the old nodes down."""

    def __init__(self, server: viser.ViserServer, sets: list[str], point_size: float):
        self.server = server
        self.nodes: list = []
        self.set_dd = server.gui.add_dropdown("set", tuple(sets), initial_value=sets[0])
        self.cloud_dd = server.gui.add_dropdown("cloud", ("raw", "clean"), initial_value="raw")
        self.size_sl = server.gui.add_slider("point size", 0.001, 0.05, 0.001, point_size)
        self.view_cb = [server.gui.add_checkbox(n, True) for n in VIEW_NAMES]
        self.info = server.gui.add_text("scene", "", disabled=True)
        self.clouds: dict[str, viser.PointCloudHandle] = {}

        self.set_dd.on_update(lambda _: self.build())
        self.cloud_dd.on_update(lambda _: self.build())
        self.size_sl.on_update(lambda _: self.apply_point_size())
        for k, cb in enumerate(self.view_cb):
            cb.on_update(lambda _, k=k: self.apply_visibility(k))
        self.build()

    def build(self) -> None:
        for node in self.nodes:
            node.remove()
        self.nodes, self.clouds = [], {}
        set_dir = GT_ROOT / self.set_dd.value
        scene = "scene_clean.npz" if self.cloud_dd.value == "clean" else "scene.npz"
        if not (set_dir / scene).exists():
            self.info.value = f"{scene} missing — run clean_scene.py"
            return
        z = np.load(set_dir / scene)
        pts, cols, vid = z["points"], z["colors"], z["view_id"]
        for k, name in enumerate(VIEW_NAMES):
            m = vid == k
            node = self.server.scene.add_point_cloud(
                f"/gt/points/{name}", points=pts[m], colors=cols[m],
                point_size=self.size_sl.value, point_shape="circle")
            node.visible = self.view_cb[k].value
            self.clouds[name] = node
            self.nodes.append(node)
        self.nodes += add_frustums(self.server, "/gt/vggt", z["extrinsics"], z["K"],
                                   z["image_size"], VGGT_COLOR)

        calib_path = set_dir / "calibration.npz"
        if calib_path.exists():
            c = np.load(calib_path, allow_pickle=True)
            self.nodes += add_frustums(self.server, "/gt/calib", c["extrinsics"], c["K"],
                                       c["image_size"], CALIB_COLOR)
        unit = "m/unit" if bool(z["metric"]) else "units (NOT metric)"
        self.info.value = (f"{len(pts)} pts | scale {float(z['scale']):.4f} {unit} "
                           f"({str(z['scale_source'])}) | calib {'yes' if calib_path.exists() else 'no'}")

    def apply_point_size(self) -> None:
        for node in self.clouds.values():
            node.point_size = self.size_sl.value

    def apply_visibility(self, k: int) -> None:
        self.clouds[VIEW_NAMES[k]].visible = self.view_cb[k].value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--point-size", type=float, default=0.006)
    args = ap.parse_args()

    sets = sorted(p.parent.name for p in GT_ROOT.glob("*/scene.npz"))
    if not sets:
        raise FileNotFoundError(f"no <set>/scene.npz under {GT_ROOT} — run reconstruct_vggt.py first")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")          # world = cam_left, OpenCV y-down
    print(f"[view_gt] {len(sets)} set(s) at http://localhost:{args.port}", flush=True)
    Viewer(server, sets, args.point_size)
    server.sleep_forever()


if __name__ == "__main__":
    main()
