"""View the fusion geometry (``geometry/``) in a viser web viewer.

Loads ``out/<stem>/geometry/`` — the static **scene** point cloud, the per-frame
**camera** frustums + centre path (from ``transform.npz``), and the per-frame
**human** points split by climber (``obj_id``). The human is stored per-frame and
deliberately unmerged (it moves), so it is **preloaded once** — every frame's
points are uploaded to the browser up front — and shown one frame at a time via a
slider / play control; playback only toggles node visibility and never re-uploads
geometry, so it stays smooth. The scene and cameras are static and shown in full.

On top of the geometry it overlays the fitted **SMPL-X bodies**, every tracked
person, in the same metric world frame, following the same frame slider as the
human points. It **prefers the refined ``sam3d/smplx_kinematic.npz``** (the
visually-guided kinematic stage — already world-frame on the 52-joint
``SMPLXMid`` body) and falls back to the raw ``sam3d/smplx_params.npz``. Each body
is one **skinned mesh** (viser does linear blend skinning in the browser): we
average ``betas`` to a single identity so the rest mesh is constant, and each
frame only updates the skeleton — far lighter than re-uploading the mesh per
frame. The trade-off is dropping the pose-corrective blendshapes (a few mm). For
the raw params ``transl`` is fit in each frame's camera, so the inverse VGGT
extrinsic is folded into the bone transforms; the refined params are already
world-frame (no fold). ``--no-smplx`` to hide.

This is the geometry counterpart to ``view_motion.py`` (which plays the SMPL
motion ``results/*.npz``); the two are kept separate.

Usage::

    uv run python scripts/view_3d.py out/<stem>/ [--port 8082] [--no-smplx]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _camera_world_pose(extrinsic: np.ndarray):
    """cam-from-world OpenCV ``[R|t]`` → ``(world position, world-from-cam wxyz)``."""
    import viser.transforms as vt

    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return -R.T @ t, vt.SO3.from_matrix(R.T).wxyz


# Per-person body colours (RGB 0-255), matching the stage overlay palette so a
# person keeps one colour across the body/SMPL-X overlays and this viewer.
_BODY_COLORS = (
    (166, 189, 219),  # light blue
    (242, 153, 102),  # salmon
    (153, 217, 140),  # light green
    (204, 166, 224),  # lilac
    (242, 217, 115),  # gold
)


def _smplx_kinematic_skinned(npz_path: Path, *, device: str = "cuda"):
    """Skinning data from the world-frame kinematic SMPL-X (``smplx_kinematic.npz``).

    Same return contract as :func:`_smplx_skinned`, but the params are already in
    the world frame (``frame='world'``) on the 52-joint ``SMPLXMid`` body, so there
    is **no** camera→world fold — ``SMPLXMid.forward`` yields world-frame bone
    transforms directly. These are the refined params; preferred when present.
    """
    import torch

    bh_src = _REPO_ROOT.parent / "better_human" / "src"
    if bh_src.exists() and str(bh_src) not in sys.path:
        sys.path.insert(0, str(bh_src))
    from better_human.smpl import SMPLXMid

    model_path = _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mid = SMPLXMid(model_path=str(model_path)).to(dev).eval()

    d = np.load(npz_path, allow_pickle=True)
    q_all = d["q"]                          # (P, N, 211)
    betas_all = d["betas"]                  # (P, 10)
    valid = d["valid_mask"].astype(bool)    # (P, N)
    P, N = q_all.shape[:2]
    # Per-person global body scale (about the pelvis) the kinematic stage fit.
    scales = d["scale"] if "scale" in d.files else np.ones(P, np.float32)
    object_ids = (np.asarray(d["object_ids"], np.int32)
                  if "object_ids" in d.files else np.arange(P, dtype=np.int32))
    faces = mid.faces.cpu().numpy().astype(np.int32)

    # Top-4 renormalized skin weights (viser keeps top-4 and does not renorm).
    lbs_w = mid.lbs_weights.cpu().numpy()   # (V, 52)
    J = lbs_w.shape[1]
    rows = np.arange(lbs_w.shape[0])[:, None]
    keep = np.argsort(lbs_w, axis=1)[:, -4:]
    weights = np.zeros_like(lbs_w)
    weights[rows, keep] = lbs_w[rows, keep]
    weights = (weights / weights.sum(1, keepdims=True)).astype(np.float32)

    people: list[dict | None] = []
    with torch.no_grad():
        for p in range(P):
            vmask = valid[p]
            idx = np.flatnonzero(vmask)
            if idx.size == 0:
                people.append(None)
                continue
            betas = torch.as_tensor(betas_all[p], dtype=torch.float32, device=dev)[None]
            v_shaped, j_rest = mid.forward_shape(betas)
            B = idx.size
            q = torch.as_tensor(q_all[p][idx], dtype=torch.float32, device=dev)
            jw = mid.forward(betas.expand(B, -1), q).joints_world.tensor()  # (B, 52, 7) world
            # Apply the global scale uniformly about the pelvis: rest geometry
            # about the rest pelvis, world bone positions about the world pelvis.
            # Browser LBS then reproduces root + s·(v_posed − root) exactly.
            s = float(scales[p])
            if s != 1.0:
                root_rest = j_rest[:, :1, :].clone()
                v_shaped = root_rest + s * (v_shaped - root_rest)
                j_rest = root_rest + s * (j_rest - root_rest)
                jwp = jw[..., :3]
                root_w = jwp[:, :1, :]
                jw = torch.cat([root_w + s * (jwp - root_w), jw[..., 3:]], dim=-1)
            quat = jw[..., 3:7].cpu().numpy()                              # xyzw
            bone_wxyz = np.full((N, J, 4), np.nan, np.float32)
            bone_pos = np.full((N, J, 3), np.nan, np.float32)
            bone_wxyz[idx] = quat[..., [3, 0, 1, 2]].astype(np.float32)   # → wxyz
            bone_pos[idx] = jw[..., :3].cpu().numpy().astype(np.float32)
            people.append({
                "v_shaped": v_shaped[0].cpu().numpy().astype(np.float32),
                "j_rest": j_rest[0].cpu().numpy().astype(np.float32),
                "weights": weights,
                "bone_wxyz": bone_wxyz,
                "bone_pos": bone_pos,
                "valid": vmask,
            })
    return faces, object_ids, people


def _smplx_skinned(pred: Path, extr: np.ndarray, *, device: str = "cuda"):
    """Build per-person LBS skinning data from stored SMPL-X params.

    Reads ``sam3d/smplx_params.npz`` (every tracked person, stacked ``(P, N, …)``;
    legacy single-person ``(N, …)`` is promoted to ``P=1``). For each person we
    average ``betas`` over its valid frames into **one identity**, so the rest mesh
    (shaped template + skin weights) is constant across the sequence; only the
    per-frame skeleton moves. This lets viser do linear blend skinning in the
    browser — each frame ships ~55 bone transforms instead of ~10k vertices. The
    cost is dropping the pose-corrective blendshapes (a few mm; ``view_motion`` /
    the overlays keep them).

    Each body's ``transl`` was fit in its frame's camera, so we fold the inverse
    VGGT extrinsic (``world_from_cam``) into the per-frame bone transforms — the
    bodies end up in the same metric world frame as the scene/human points.

    Returns ``(faces (F, 3) int32, object_ids (P,), people)`` or ``None`` when no
    ``smplx_params.npz`` exists. ``people[p]`` is ``None`` for a person with no
    valid frames, else a dict with ``v_shaped (V, 3)``, ``j_rest (J, 3)``,
    ``weights (V, J)`` (the rest pose passed to ``add_mesh_skinned``) and per-frame
    ``bone_wxyz (N, J, 4)`` / ``bone_pos (N, J, 3)`` world transforms (NaN where
    invalid) plus ``valid (N,)``.
    """
    # Prefer the refined world-frame kinematic params when present.
    kin = pred / "sam3d" / "smplx_kinematic.npz"
    if kin.exists():
        return _smplx_kinematic_skinned(kin, device=device)

    path = pred / "sam3d" / "smplx_params.npz"
    if not path.exists():
        return None

    import warnings

    import torch
    import viser.transforms as vt
    from smplx.lbs import batch_rodrigues, blend_shapes, vertices2joints

    sx = np.load(path)
    pose_keys = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                 "left_hand_pose", "right_hand_pose")
    keys = ("betas", "transl") + pose_keys
    params = {k: sx[k] for k in keys}
    if params["betas"].ndim == 2:  # legacy single-person (N, …) → (1, N, …)
        params = {k: v[None] for k, v in params.items()}
    valid = sx["valid_mask"]
    valid = (valid[None] if valid.ndim == 1 else valid).astype(bool)

    P, N = params["betas"].shape[:2]
    object_ids = (np.asarray(sx["object_ids"], np.int32)
                  if "object_ids" in sx.files else np.arange(P, dtype=np.int32))
    gender = str(sx["gender"]) if "gender" in sx.files else "neutral"
    model_type = str(sx["model_type"]) if "model_type" in sx.files else "smplx"
    num_betas = int(sx["num_betas"]) if "num_betas" in sx.files else params["betas"].shape[-1]
    num_expr = (int(sx["num_expression_coeffs"]) if "num_expression_coeffs" in sx.files
                else 10)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    models_dir = _REPO_ROOT.parent / "better_human" / "models"

    import smplx as smplx_lib

    # One model for its static buffers (template/J_regressor/lbs_weights/tree);
    # the FK below uses smplx.lbs directly, so batch_size is irrelevant.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed")
        model = smplx_lib.create(
            model_path=str(models_dir), model_type=model_type, gender=gender,
            use_pca=False, flat_hand_mean=False, num_betas=num_betas,
            num_expression_coeffs=num_expr, batch_size=1).to(dev)
    faces = np.asarray(model.faces, dtype=np.int32)
    parents = model.parents.long()
    j_regressor = model.J_regressor

    # viser keeps only each vertex's top-4 bone weights and does NOT renormalize
    # them. SMPL-X spreads weight over up to ~8 bones, so the dropped tail leaves
    # blend-region vertices summing to <1; because we bake the metre-scale world
    # translation into the bones, that deficit flings vertices out as spikes. So
    # pre-restrict to the top 4 and renormalize to sum 1 (matches what viser uses).
    lbs_w = model.lbs_weights.cpu().numpy()           # (V, J)
    J = lbs_w.shape[1]
    keep = np.argsort(lbs_w, axis=1)[:, -4:]
    weights = np.zeros_like(lbs_w)
    weights[np.arange(lbs_w.shape[0])[:, None], keep] = lbs_w[
        np.arange(lbs_w.shape[0])[:, None], keep]
    weights = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)

    # world_from_cam per frame: T_wc = [[Rᵀ, -Rᵀt], [0, 1]].
    M = min(N, len(extr))
    Rcw = extr[:M, :3, :3].astype(np.float32)
    Twc = np.zeros((M, 4, 4), np.float32)
    Twc[:, :3, :3] = Rcw.transpose(0, 2, 1)
    Twc[:, :3, 3] = -np.einsum("nji,nj->ni", Rcw, extr[:M, :3, 3].astype(np.float32))
    Twc[:, 3, 3] = 1.0
    Twc_t = torch.from_numpy(Twc).to(dev)

    def make_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        T = torch.zeros(R.shape[0], 4, 4, device=dev)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        T[:, 3, 3] = 1.0
        return T

    people: list[dict | None] = []
    with torch.no_grad():
        for p in range(P):
            vmask = valid[p, :M]
            idx = np.flatnonzero(vmask)
            if idx.size == 0:
                people.append(None)
                continue

            # Single averaged identity → constant rest mesh + joints.
            betas = torch.from_numpy(params["betas"][p][idx].mean(0, keepdims=True)).to(dev)
            v_shaped = (model.v_template[None] + blend_shapes(betas, model.shapedirs))[0]  # (V,3)
            j_rest = vertices2joints(j_regressor, v_shaped[None])[0]                        # (J,3)
            rel = j_rest.clone()
            rel[1:] = j_rest[1:] - j_rest[parents[1:]]                                      # bone vectors

            # Per-frame forward kinematics → global joint transforms (camera frame).
            B = idx.size
            pose = np.concatenate([params[k][p][idx] for k in pose_keys], axis=1)           # (B, J*3)
            rot = batch_rodrigues(torch.from_numpy(pose).to(dev).reshape(-1, 3)).reshape(B, J, 3, 3)
            chain = [make_T(rot[:, 0], rel[0][None].expand(B, 3))]
            for j in range(1, J):
                chain.append(chain[parents[j].item()] @ make_T(rot[:, j], rel[j][None].expand(B, 3)))
            G = torch.stack(chain, dim=1)                                                   # (B,J,4,4)
            transl = torch.from_numpy(params["transl"][p][idx]).to(dev)
            G[:, :, :3, 3] = G[:, :, :3, 3] + transl[:, None, :]                            # add transl
            Gw = Twc_t[idx][:, None] @ G                                                    # → world frame

            wxyz = vt.SO3.from_matrix(
                Gw[:, :, :3, :3].reshape(-1, 3, 3).cpu().numpy()).wxyz.reshape(B, J, 4)
            bone_wxyz = np.full((N, J, 4), np.nan, np.float32)
            bone_pos = np.full((N, J, 3), np.nan, np.float32)
            bone_wxyz[idx] = wxyz.astype(np.float32)
            bone_pos[idx] = Gw[:, :, :3, 3].cpu().numpy().astype(np.float32)

            people.append({
                "v_shaped": v_shaped.cpu().numpy().astype(np.float32),
                "j_rest": j_rest.cpu().numpy().astype(np.float32),
                "weights": weights,
                "bone_wxyz": bone_wxyz,
                "bone_pos": bone_pos,
                "valid": vmask,
            })
    return faces, object_ids, people


def view_3d(pred: Path, *, port: int = 8082, point_size: float = 0.02,
            camera_scale: float = 0.15, camera_stride: int = 1, up: str = "-y",
            show_smplx: bool = True, device: str = "cuda") -> None:
    import viser

    geo = pred / "geometry"
    if not (geo / "scene.npz").exists():
        raise FileNotFoundError(f"no fusion output under {geo} — run the 'fuse' stage first")
    transform = np.load(geo / "transform.npz")
    scene = np.load(geo / "scene.npz")
    human = np.load(geo / "human.npz")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction(up)  # VGGT world is OpenCV y-down → up ≈ -y
    server.scene.add_point_cloud("/scene", scene["points"], scene["colors"], point_size=point_size)

    # Cameras — a frustum per (strided) frame + the full centre path.
    extr = transform["extrinsics"]
    aspect = float(transform["image_width"]) / float(transform["image_height"])
    fov_y = np.radians(transform["fov_y_deg"].astype(np.float64))
    for k in range(0, len(extr), camera_stride):
        pos, wxyz = _camera_world_pose(extr[k])
        server.scene.add_camera_frustum(
            f"/cameras/cam_{k:04d}", fov=float(fov_y[k]), aspect=aspect,
            scale=camera_scale, color=(90, 160, 250), wxyz=wxyz, position=pos)
    centers = np.stack([_camera_world_pose(e)[0] for e in extr]).astype(np.float32)
    server.scene.add_spline_catmull_rom("/cameras/path", centers, color=(255, 140, 0), line_width=2.0)

    # Human — per-frame, split by climber. Preloaded once: every frame's points
    # (for every climber) are uploaded to the browser up front as one hidden node
    # each, grouped under a per-climber parent so each climber stays toggleable.
    # Playback then only flips node visibility — no per-frame geometry upload, so
    # no GPU buffer re-allocation / GC churn (re-adding the cloud every frame was
    # the playback-lag culprit). All human points fit in memory many times over.
    pts, col = human["points"], human["colors"]
    fid, obj = human["frame_id"], human["obj_id"]
    obj_ids = [int(o) for o in np.unique(obj)]
    n_frames = len(extr)

    obj_masks = {o: (obj == o) for o in obj_ids}
    for o in obj_ids:  # toggleable per-climber group (hides all its frames at once)
        server.scene.add_frame(f"/human/climber_{o:02d}", show_axes=False)
    human_nodes: list[list[object]] = [[] for _ in range(n_frames)]
    for f in range(n_frames):
        on = fid == f
        for o in obj_ids:
            sel = on & obj_masks[o]
            if sel.any():
                human_nodes[f].append(server.scene.add_point_cloud(
                    f"/human/climber_{o:02d}/frame_{f:04d}",
                    pts[sel], col[sel], point_size=point_size, visible=False))

    # SMPL-X bodies — one skinned mesh per tracked person, in the same world frame
    # and on the same slider as the human points. Added once at the rest pose;
    # each frame only updates the bone transforms (browser-side LBS), so playback
    # stays light. Folded camera→world transform means the body tracks in metric.
    smplx_data = _smplx_skinned(pred, extr, device=device) if show_smplx else None
    body_handles: list[tuple[object, dict] | None] = []
    show_bodies = None
    if smplx_data is not None:
        sx_faces, sx_obj, people = smplx_data
        for p, person in enumerate(people):
            if person is None:
                body_handles.append(None)
                continue
            ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32),
                            (person["j_rest"].shape[0], 1))
            handle = server.scene.add_mesh_skinned(
                f"/smplx/person_{int(sx_obj[p]):02d}", person["v_shaped"], sx_faces,
                bone_wxyzs=ident, bone_positions=person["j_rest"],
                skin_weights=person["weights"],
                color=_BODY_COLORS[p % len(_BODY_COLORS)])
            body_handles.append((handle, person))

        def show_bodies(f: int) -> None:  # noqa: F811
            with server.atomic():
                for hp in body_handles:
                    if hp is None:
                        continue
                    handle, person = hp
                    if f < len(person["valid"]) and person["valid"][f]:
                        handle.visible = True
                        bw, bp = person["bone_wxyz"][f], person["bone_pos"][f]
                        for j, bone in enumerate(handle.bones):
                            bone.wxyz, bone.position = bw[j], bp[j]
                    else:
                        handle.visible = False

    # One state-guarded render per frame: flip the previous frame's human nodes
    # off and the new frame's on (preloaded, so this is pure visibility), then
    # pose the bodies. The guard makes repeat / paused renders no-ops.
    state = {"frame": -1}  # currently-rendered frame (-1 = nothing rendered yet)

    def render(f: int) -> None:
        prev = state["frame"]
        if f == prev:
            return
        if len(pts):
            if prev >= 0:
                for h in human_nodes[prev]:
                    h.visible = False  # type: ignore[attr-defined]
            for h in human_nodes[f]:
                h.visible = True  # type: ignore[attr-defined]
        if show_bodies is not None:
            show_bodies(f)
        state["frame"] = f

    frame = play = play_fps = None
    if len(pts) or show_bodies is not None:
        frame = server.gui.add_slider("frame", 0, n_frames - 1, 1, 0)
        play = server.gui.add_checkbox("play", False)
        play_fps = server.gui.add_slider("play fps", 1, 60, 1, 30)
        frame.on_update(lambda _: render(int(frame.value)))
        render(0)

    bodies_msg = ""
    if smplx_data is not None:
        shown = [int(o) for h, o in zip(body_handles, smplx_data[1]) if h is not None]
        bodies_msg = f"bodies {len(shown)} {shown} (LBS) | "
    print(f"[view_3d] {pred.name}: scene {len(scene['points'])} pts | {len(extr)} cameras | "
          f"human {len(pts)} pts, climbers {obj_ids} | {bodies_msg}http://localhost:{port}")

    while True:
        if play is not None and play.value:
            frame.value = (int(frame.value) + 1) % n_frames
            render(int(frame.value))
            time.sleep(1.0 / float(play_fps.value))
        else:
            time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("predictions", type=Path, help="clip dir, e.g. out/<stem>/")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--point-size", type=float, default=0.02, help="point radius, metres (default 0.02)")
    p.add_argument("--camera-scale", type=float, default=0.15, help="frustum size, metres (default 0.15)")
    p.add_argument("--camera-stride", type=int, default=1, help="show every Nth camera frustum (default 1)")
    p.add_argument("--up", default="-y", choices=["+x", "+y", "+z", "-x", "-y", "-z"],
                   help="viewer up direction (default -y, OpenCV world)")
    p.add_argument("--smplx", action=argparse.BooleanOptionalAction, default=True,
                   help="overlay fitted SMPL-X bodies from sam3d/smplx_params.npz "
                        "(default: ON; --no-smplx to hide)")
    p.add_argument("--device", default="cuda",
                   help="device for the SMPL-X forward (default cuda; falls back to cpu)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    view_3d(args.predictions, port=args.port, point_size=args.point_size,
            camera_scale=args.camera_scale, camera_stride=args.camera_stride, up=args.up,
            show_smplx=args.smplx, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
