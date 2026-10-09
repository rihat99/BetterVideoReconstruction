"""The viewer app end to end on a fake viser server: a synthetic ``out/<stem>/`` clip through
loading → BetterRobot layers → sidebar, with a hidden invalid frame, source switching and the pager."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest
from better_robot.viewer import Viewer

from bvr.body import load_body
from bvr.viewer.app import App
from bvr.viewer.layers import _FORCE_SCALE, _TORQUE_SCALE
from bvr.viewer.loading import VideoFrames, load_clip_data


class FakeHandle:
    """A scene node or GUI control: keeps its kwargs, its last written attributes and its callbacks."""

    def __init__(self, name: str, kwargs: dict) -> None:
        self.name, self.kwargs, self.callbacks, self.removed = name, kwargs, [], False
        self.visible, self.value = kwargs.get("visible", True), kwargs.get("initial_value")
        if "bone_wxyzs" in kwargs:            # a skinned mesh: one handle per bone
            self.bones = [FakeHandle(f"{name}/bone{i}", {}) for i in range(len(kwargs["bone_wxyzs"]))]

    def remove(self) -> None:
        self.removed = True

    def on_update(self, callback) -> None:
        self.callbacks.append(callback)

    on_click = on_update

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


class FakeApi:
    """``server.scene`` / ``server.gui``: every ``add_*`` returns a :class:`FakeHandle`."""

    def __init__(self) -> None:
        self.nodes: dict[str, FakeHandle] = {}

    def __getattr__(self, method: str):
        if method.startswith(("set_", "configure_")):
            return lambda *args, **kwargs: None
        if not method.startswith("add_"):
            raise AttributeError(method)

        def add(*args, **kwargs) -> FakeHandle:
            name = args[0] if args and isinstance(args[0], str) else method
            handle = self.nodes[name] = FakeHandle(name, kwargs)
            return handle

        return add


class FakeServer:
    def __init__(self) -> None:
        self.scene, self.gui = FakeApi(), FakeApi()

    def get_port(self) -> int:
        return 0

    @contextlib.contextmanager
    def atomic(self):
        yield

    def on_client_connect(self, callback) -> None:
        self.connect = callback

    def get_clients(self) -> dict:
        return {}


def _write_clip(root: Path, body, n_frames: int) -> Path:
    """A tiny clip: scene, moving camera, human points (none on frame 2), gravity + ground, and three
    body sources — kinematics (frame 3 invalid, stored as zeros), its contacts on two frames, and kindyn."""
    clip = root / f"clip_{n_frames}"
    for sub in ("geometry", "human_optim", "geocalib"):
        (clip / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    T = n_frames
    extr = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    extr[:, 2, 3] = np.linspace(3.0, 3.5, T)
    np.savez(clip / "geometry" / "transform.npz", extrinsics=extr, fov_y_deg=np.full(T, 50.0, np.float32),
             image_width=np.int32(640), image_height=np.int32(480), fps=np.float32(30.0),
             frame_indices=np.arange(T, dtype=np.int32), metric=np.bool_(True))
    np.savez(clip / "geometry" / "scene.npz", points=rng.normal(size=(50, 3)).astype(np.float32),
             colors=np.full((50, 3), 128, np.uint8), confidence=np.ones(50, np.float32),
             normals=np.tile(np.float32([0, -1, 0]), (50, 1)), voxel_size=np.float32(0.02))
    frame_id = np.repeat(np.arange(T), 4)
    frame_id = frame_id[frame_id != 2].astype(np.int32)
    np.savez(clip / "geometry" / "human.npz", points=rng.normal(size=(len(frame_id), 3)).astype(np.float32),
             colors=np.full((len(frame_id), 3), 200, np.uint8), frame_id=frame_id,
             obj_id=np.zeros(len(frame_id), np.int32))
    np.savez(clip / "geocalib" / "gravity.npz", gravity_world=np.float32([0, 1, 0]), source=np.str_("geocalib"),
             ground_accepted=np.bool_(True), ground_centroid=np.float32([0, 1, 3]), ground_normal=np.float32([0, -1, 0]),
             ground_extent=np.float32(2.0), ground_angle_deg=np.float32(1.0), ground_reject=np.str_(""))

    q = np.tile(np.zeros(body.nq, np.float32), (1, T, 1))
    q[..., 6::4] = 1.0                                    # every joint quaternion is the identity (xyzw)
    q[..., :3] = [0.0, 0.0, 3.0]
    valid = np.ones((1, T), bool)
    valid[0, 3] = False
    q[0, 3] = 0.0                                         # an untracked frame: all-zero quaternions
    common = dict(q=q, betas=np.zeros((1, 10), np.float32), valid_mask=valid, object_ids=np.int32([0]),
                  joint_names=np.array(body.joint_names), parents=body.parent_tree[0].numpy().astype(np.int32))
    names = list(body.contact_frame_names)
    np.savez(clip / "human_optim" / "kinematic_1.npz", **common)
    subset = np.zeros((1, T, 2), bool)
    subset[0, ::2, 0] = True                              # the first named frame touches on even frames
    np.savez(clip / "human_optim" / "contacts_1.npz", contact_label_schema=np.int32(2), frame_contact=subset,
             contact_frame_names=np.array(names[:2]), object_ids=np.int32([0]))
    J, F = len(body.joint_names), len(names)
    np.savez(clip / "human_optim" / "kindyn_1.npz", **common, frame_contact=np.ones((1, T, F), bool),
             contact_frame_names=np.array(names), contact_forces_world=np.ones((1, T, J, 3), np.float32),
             torque_mag=rng.uniform(size=(1, T, J)).astype(np.float32),
             frame_forces=np.ones((1, T, F, 3), np.float32), base_wrench=np.ones((1, T, 6), np.float32))
    return clip


@pytest.fixture(scope="module")
def body():
    return load_body()


def _items(app: App, source: str, kind: str):
    return [item for item in app.items if item.source == source and item.kind == kind]


def test_app_draws_every_source_and_hides_invalid_frames(tmp_path, body):
    clip = _write_clip(tmp_path, body, 6)
    viewer = Viewer(server=FakeServer())
    app = App(viewer, [clip], body=body, video=False)

    assert list(app.data.bodies) == ["kinematics", "kindyn_1"]
    assert app.source.value == "kinematics" and viewer.timeline.nsteps == 6
    assert all(np.isfinite(src.q).all() for src in app.data.bodies.values())
    mesh = _items(app, "kinematics", "body")[0].layer
    assert mesh.visible and not _items(app, "kindyn_1", "body")[0].layer.visible

    viewer.show(3)                                        # the untracked frame hides the person ...
    assert not mesh._mesh.visible
    viewer.show(4)
    assert mesh._mesh.visible
    human = next(item.layer for item in app.items if item.kind == "human")
    app.kinds["human"].value = True
    app._apply()
    viewer.show(2)                                        # ... and a frame without depth points
    assert not human._cloud.visible

    # Contacts of a subset artifact land on the right frames, red on even steps.
    spheres = _items(app, "kinematics", "contact_frames")[0].layer
    assert spheres._colors[0, 0, 0] > spheres._colors[0, 0, 1] and spheres._colors[1, 0, 1] > spheres._colors[1, 0, 0]
    assert spheres._colors[0, 2, 1] > spheres._colors[0, 2, 0]       # a frame the artifact does not cover is free

    # The force checkboxes appear only with the dynamics body.
    assert not app.kinds["joint_forces"].visible
    app.source.value = "kindyn_1"
    app._apply()
    assert app.kinds["joint_forces"].visible and not mesh.visible
    assert all(item.layer.visible == app.kinds[item.kind].value for item in app.items if item.source == "kindyn_1")


def test_pager_swaps_groups_and_shrinks_the_timeline(tmp_path, body):
    long, short = _write_clip(tmp_path, body, 8), _write_clip(tmp_path, body, 5)
    viewer = Viewer(server=FakeServer())
    app = App(viewer, [long, short], body=body, video=False)
    first = list(app.items)
    app.load(short)

    assert viewer.timeline.nsteps == 5
    assert all(item.layer not in viewer.layers for item in first)
    assert len(viewer.layers) == len(app.items)


def test_pager_resets_the_video_pane_and_keeps_the_clip_when_a_read_fails(tmp_path, body):
    rig, plain, broken = _write_clip(tmp_path, body, 8), _write_clip(tmp_path, body, 5), tmp_path / "broken"
    broken.mkdir()

    def read(clip, **opts):
        data = load_clip_data(clip, **opts)
        if clip == rig:                                   # two views whose footage is missing: black frames
            data.videos = {v: VideoFrames(clip / f"{v}.mp4", np.arange(data.n_frames)) for v in ("left", "right")}
        return data

    viewer = Viewer(server=FakeServer())
    app = App(viewer, [rig, plain, broken], body=body, read=read)
    assert app.view_menu.visible and app.video_folder.visible and list(app.panels) == ["left", "right"]
    app.load(plain)                                       # no video: no stale view menu or empty folder
    assert not app.panels and not app.view_menu.visible and not app.video_folder.visible
    shown = list(app.items)
    app.load(broken)                                      # no geometry/: the read fails, the clip stays
    assert app.data.name == plain.name and app.items == shown and viewer.timeline.nsteps == 5
    assert all(item.layer in viewer.layers for item in shown)


def test_camera_and_root_wrench_follow_the_stored_conventions(tmp_path, body):
    """Hand-computed expectations for a camera and a root both turned 90 degrees about z."""
    clip = _write_clip(tmp_path, body, 6)
    s = np.sqrt(0.5)
    # cam-from-world (OpenCV) R = Rz(+90), t = (1, 2, 3): the camera sits at -R^T t = (-2, 1, -3), turned Rz(-90).
    transform = dict(np.load(clip / "geometry" / "transform.npz"))
    transform["extrinsics"][:] = np.float32([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    np.savez(clip / "geometry" / "transform.npz", **transform)
    # The root quaternion is xyzw, Rz(+90): a root-local force along +x points along world +y, a torque along +y
    # along world -x.
    kindyn = dict(np.load(clip / "human_optim" / "kindyn_1.npz"))
    kindyn["q"][..., 3:7] = [0.0, 0.0, s, s]
    kindyn["base_wrench"][:] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    np.savez(clip / "human_optim" / "kindyn_1.npz", **kindyn)

    app = App(Viewer(server=FakeServer()), [clip], body=body, video=False)
    camera = next(item.layer for item in app.items if item.layer.name == "camera")._frustum
    assert np.allclose(camera.position, [-2.0, 1.0, -3.0], atol=1e-6)
    wxyz = np.asarray(camera.wxyz)                        # viser's order
    assert np.allclose(wxyz * np.sign(wxyz[0]), [s, 0.0, 0.0, -s], atol=1e-6)
    wrench = _items(app, "kindyn_1", "root_wrench")[0].layer.vectors[0]
    assert np.allclose(wrench, [[0.0, _FORCE_SCALE, 0.0], [-_TORQUE_SCALE, 0.0, 0.0]], atol=1e-7)
