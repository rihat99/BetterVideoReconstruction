"""Offline scoring of the drift-gated contact READOUT vs manual annotation — the GATE.

Replicates ``estimate_contacts``'s readout (pose → scene SDF → guarded per-joint min
signed distance → drift/stillness gate → temporal hysteresis) on the optimized
``kinematic_1`` bodies and scores it against the manual per-joint annotation
``data/<stem>.npz``, sweeping the readout's two decisive knobs:

  * ``close_thresh_m`` — the SDF "just outside" distance counted as contact;
  * ``still_m`` — the net-drift scale of the stillness gate (``still_m = inf`` is the
    NO-GATE column, i.e. the pure SDF baseline the round-1 work started from).

The self-confirmation trap (``docs/UPDATE.md`` §3) is why we report **precision, recall
AND the false-positive rate on annotated no-contact joint-frames** per clip (per coarse
joint group + overall): recall is exactly the number a contact-attraction pull inflates
for free, so precision + FPR are the honest signal. The GATE the user decides on: an
operating point whose precision is at or above the round-1 detector's while recall is
well above it (the hard speed veto was the recall killer — the windowed drift gate
should give most of it back).

Scene is loaded exactly as ``estimate_contacts`` loads it (``[contacts]`` conf gate +
PCA normals), so the grid scores the identical readout the stage writes.

    uv run python scripts/prototype_auto_contacts.py --predictions out/boulder_1/ out/boulder_3/ [--device cuda]
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import tomllib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.geometry.normals import estimate_point_normals  # noqa: E402
from tools.human_optim import (  # noqa: E402
    drift_gated_contact, joint_drift, joint_guarded_min_sdf, scene_sdf,
    vertex_to_joint, world_from_cam,
)
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.human_optim.stages import _ANN_TO_SMPLX, resolve_annotation_path  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_DEFAULT_SMPLX_MODEL = _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
_COARSE = list(_ANN_TO_SMPLX.keys())   # the 10 annotated coarse joints

# The readout grid. still_m = inf ⇒ no drift gate = the pure SDF baseline (round 1's start).
_CLOSE_GRID = (0.02, 0.03)
_STILL_GRID = (0.02, 0.025, 0.03, 0.05, float("inf"))
# Round-1 detector numbers (from docs/UPDATE_AGENT.md §1), printed as a static reference.
_ROUND1_DETECTOR = "precision 0.83–0.97, recall 0.19–0.65, FPR 0.01–0.21 (hard speed veto)"


# ─────────────────────────────── scoring ─────────────────────────────────

def _members(mid_names: list[str], coarse: str) -> list[int]:
    idx = {n: i for i, n in enumerate(mid_names)}
    return [idx[j] for j in _ANN_TO_SMPLX[coarse] if j in idx]


def _coarse_pred(member_bool: np.ndarray) -> np.ndarray:
    """(N,) bool: a coarse joint is predicted in contact iff ANY member joint is."""
    return member_bool.any(axis=1) if member_bool.size else np.zeros(0, bool)


def _rates(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, fpr=fpr,
               n_pos=tp + fn, n_neg=fp + tn)


def _score(pred_coarse: dict[str, np.ndarray], ann: np.ndarray, ann_names: list[str],
           valid: np.ndarray) -> dict:
    """Confusion counts per coarse joint over frames where the annotation is 0/1 and valid.

    ``pred_coarse[c]`` is (N,) bool; ``ann`` is (J_ann, N) tri-state; returns per-joint
    and 'overall' dicts of tp/fp/fn/tn + precision/recall/fpr.
    """
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    out = {}
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for c in _COARSE:
        if c not in ann_idx or c not in pred_coarse:
            continue
        a = ann[ann_idx[c]]                              # (N,) tri-state
        p = pred_coarse[c]                               # (N,) bool
        m = valid & ((a == 0) | (a == 1))                # scored frames
        if not m.any():
            continue
        av, pv = (a[m] == 1), p[m]
        tp = int((pv & av).sum()); fp = int((pv & ~av).sum())
        fn = int((~pv & av).sum()); tn = int((~pv & ~av).sum())
        out[c] = _rates(tp, fp, fn, tn)
        for k, v in zip(("tp", "fp", "fn", "tn"), (tp, fp, fn, tn)):
            tot[k] += v
    out["overall"] = _rates(**tot)
    return out


def _fmt(r: dict) -> str:
    def g(x):
        return "  n/a" if x != x else f"{x:5.3f}"
    return (f"P={g(r['precision'])} R={g(r['recall'])} FPR={g(r['fpr'])}  "
            f"(pos={r['n_pos']:4d} neg={r['n_neg']:4d})")


def _still_tag(s: float) -> str:
    return "nogate" if s == float("inf") else f"{s:.3f}"


# ─────────────────────────────── readout ─────────────────────────────────

@torch.no_grad()
def _pose_world(mid, betas: torch.Tensor, q: torch.Tensor, scale: float, *, chunk=64):
    """Pose all frames → ``(joints (N,J,3), vertices (N,V,3))`` world (scale about pelvis).

    Identical to ``estimate_contacts._pose`` so the grid scores what the stage writes.
    """
    N = q.shape[0]
    jws, vws = [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out = mid.forward(betas[None].expand(e - s, -1), normalize_q(q[s:e], mid.num_joints))
        jw = out.joints_world.tensor()[..., :3]
        vw = out.vertices
        root = jw[:, :1, :]
        jws.append(root + scale * (jw - root))
        vws.append(root + scale * (vw - root))
    return torch.cat(jws, 0), torch.cat(vws, 0)


@torch.no_grad()
def _readout_signals(mid, betas, q, scale, scene_pts, scene_normals, scene_conf, *,
                     fps: float, cc: dict, cg: dict):
    """Grid-invariant readout signals for one person: ``(guarded_min_sdf, drift)`` (T,J).

    Everything the (close_thresh_m, still_m) sweep does NOT change — the pose, the scene
    SDF, the guarded per-joint min, and the net window drift — computed once.
    """
    J = mid.num_joints
    v_rest, j_rest = mid.forward_shape(betas[None])
    v2j = vertex_to_joint(v_rest[0], j_rest[0])                       # (V,) per person
    jw, vw = _pose_world(mid, betas, q, scale)
    sdf, dmin, conf = scene_sdf(vw, scene_pts, scene_normals, scene_conf,
                                prune_margin_m=float(cc["prune_margin_m"]))
    gsdf = joint_guarded_min_sdf(sdf, dmin, conf, v2j, J,
                                 max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]))
    drift = joint_drift(jw, fps=fps, window_s=float(cg["window_s"]))
    return gsdf, drift


def _load_scene_readout(pred: Path, cc: dict, cam_centers: torch.Tensor, device: str):
    """Scene cloud + PCA normals, gated exactly as ``estimate_contacts`` does."""
    sc = dict(np.load(pred / "geometry" / "scene.npz", allow_pickle=True))
    scene_pts = torch.as_tensor(sc["points"], dtype=torch.float32, device=device)
    scene_conf = torch.as_tensor(sc["confidence"], dtype=torch.float32, device=device)
    q = float(cc.get("conf_drop_q", 0.0))
    if q > 0.0 and scene_conf.numel():
        keep = scene_conf >= torch.quantile(scene_conf, q)
        scene_pts, scene_conf = scene_pts[keep], scene_conf[keep]
    scene_normals = estimate_point_normals(scene_pts, cam_centers, k=int(cc["normal_k"]))
    return scene_pts, scene_normals, scene_conf


def _load_contact_cfg(path: Path) -> tuple[dict, dict]:
    """``([contacts], [contact_gate])`` with the same defaults as ``estimate_contacts``."""
    cfg = tomllib.load(path.open("rb")) if path.exists() else {}
    cc = cfg.get("contacts", {})
    cc.setdefault("max_dist_m", 0.10); cc.setdefault("conf_thresh", 0.1)
    cc.setdefault("conf_drop_q", 0.1); cc.setdefault("normal_k", 16)
    cc.setdefault("prune_margin_m", 0.10)
    cc.setdefault("exit_thresh_m", 0.035); cc.setdefault("min_frames", 3)
    cg = cfg.get("contact_gate", {})
    cg.setdefault("window_s", 0.35); cg.setdefault("still_m", 0.025)
    return cc, cg


def run_one(pred_dir: str, config_path: Path, device: str, source=None,
            body_path: Path | None = None) -> list[dict]:
    """Score the readout grid on one clip's ``kinematic_1`` body (or ``body_path``).

    Returns a per-person list of ``{clip, oid, valid, grid}`` where ``grid`` maps
    ``(close_thresh_m, still_m) -> _score(...)`` (per-coarse-joint + overall confusion).
    """
    pred = Path(pred_dir)
    cc, cg = _load_contact_cfg(config_path)
    bpath = Path(body_path) if body_path is not None else paths.body_path(pred, "kinematic_1")
    kin = dict(np.load(bpath, allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    N, P = int(kin["num_frames"]), int(kin["num_people"])
    object_ids = np.asarray(kin["object_ids"], np.int32)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    cam_centers = world_from_cam(extr)[:, :3, 3]
    scene_pts, scene_normals, scene_conf = _load_scene_readout(pred, cc, cam_centers, device)

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    mid_names = list(mid.joint_names)
    member_cols = {c: _members(mid_names, c) for c in _COARSE}
    exit_default = float(cc["exit_thresh_m"]); min_frames = int(cc["min_frames"])

    ann_path = resolve_annotation_path(source or pred, None)
    ann_all = ann_names = ann_oids = None
    if ann_path.exists():
        a = dict(np.load(ann_path, allow_pickle=True))
        ann_all = np.asarray(a["contacts"])                       # (Pa, J_ann, N)
        ann_names = [str(x) for x in a["joint_names"]]
        ann_oids = [int(x) for x in np.asarray(a["object_ids"]).ravel()]

    results = []
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(kin["valid_mask"][p], bool)
        if ann_all is None or oid not in ann_oids:
            continue                                              # score only annotated people
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        scale = float(kin["scale"][p]) if "scale" in kin else 1.0
        gsdf, drift = _readout_signals(mid, betas, q, scale, scene_pts, scene_normals, scene_conf,
                                       fps=fps, cc=cc, cg=cg)
        ann = ann_all[ann_oids.index(oid)]                        # (J_ann, N)
        grid = {}
        for close in _CLOSE_GRID:
            exit_below = max(exit_default, close)
            for still in _STILL_GRID:
                jc = drift_gated_contact(gsdf, drift, close_thresh_m=close, exit_thresh_m=exit_below,
                                         min_frames=min_frames, still_m=still).cpu().numpy()
                jc[~vmask] = False
                pred_coarse = {c: _coarse_pred(jc[:, member_cols[c]]) for c in _COARSE}
                grid[(close, still)] = _score(pred_coarse, ann, ann_names, vmask)
        results.append(dict(clip=pred.name, oid=oid, valid=int(vmask.sum()), n=N, grid=grid))
    return results


# ─────────────────────────────── reporting ───────────────────────────────

def _print_clip_grid(res: dict) -> None:
    print(f"\n=== {res['clip']}  oid={res['oid']}  ({res['valid']}/{res['n']} valid) ===")
    for close in _CLOSE_GRID:
        print(f"  close_thresh={close * 100:.0f}cm")
        for still in _STILL_GRID:
            r = res["grid"][(close, still)]["overall"]
            print(f"    still={_still_tag(still):>6s}  {_fmt(r)}")


def _pool(all_res: list[dict], close: float, still: float) -> dict:
    """Pool overall confusion across all people/clips for one grid cell."""
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for res in all_res:
        o = res["grid"][(close, still)]["overall"]
        for k in tot:
            tot[k] += o[k]
    return _rates(**tot)


def _print_pooled(all_res: list[dict]) -> None:
    print("\n=== POOLED across clips (overall P / R / FPR per grid cell) ===")
    print(f"  round-1 detector reference: {_ROUND1_DETECTOR}")
    for close in _CLOSE_GRID:
        print(f"  close_thresh={close * 100:.0f}cm")
        for still in _STILL_GRID:
            print(f"    still={_still_tag(still):>6s}  {_fmt(_pool(all_res, close, still))}")


def _print_per_joint(all_res: list[dict], close: float, still: float) -> None:
    print(f"\n=== PER-COARSE-JOINT at close={close * 100:.0f}cm still={_still_tag(still)} ===")
    for res in all_res:
        print(f"  {res['clip']} oid={res['oid']}")
        sc = res["grid"][(close, still)]
        for c in _COARSE:
            if c in sc:
                print(f"    {c:12s} {_fmt(sc[c])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", nargs="+", required=True, help="one or more clip dirs")
    ap.add_argument("--source", default=None, help="annotation override (default: data/<stem>.npz per clip)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref-close", type=float, default=0.02, help="per-joint report cell: close_thresh_m")
    ap.add_argument("--ref-still", type=float, default=0.025, help="per-joint report cell: still_m")
    args = ap.parse_args(argv)

    all_res: list[dict] = []
    for pd in args.predictions:
        all_res.extend(run_one(pd, args.config, args.device, source=args.source))
    if not all_res:
        print("no annotated people found in the given clips")
        return 1
    for res in all_res:
        _print_clip_grid(res)
    _print_per_joint(all_res, args.ref_close, args.ref_still)
    _print_pooled(all_res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
