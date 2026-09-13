"""Offline scoring of the drift-gated contact READOUT vs manual annotation — the GATE.

Replicates ``estimate_contacts``'s readout (pose → scene SDF → guarded per-joint min
signed distance → drift/stillness gate → temporal hysteresis) on the optimized
``kinematic_1`` bodies and scores it against the manual per-joint annotation
``data/<stem>.npz``, sweeping the readout's two decisive knobs:

  * ``close_thresh_m`` — the SDF "just outside" distance counted as contact;
  * ``still_m`` — the net-drift scale of the stillness gate (``still_m = inf`` is the
    NO-GATE column, i.e. the pure SDF baseline).

The self-confirmation trap is why we report **precision, recall AND the false-positive
rate on annotated no-contact joint-frames** per clip (per coarse joint group + overall):
recall is exactly the number a contact-attraction pull inflates for free, so precision +
FPR are the honest signal. The GATE the user decides on: a strong-precision operating
point whose recall sits well above the pure-SDF baseline.

Scene is loaded exactly as ``estimate_contacts`` loads it (``[contacts]`` conf gate +
PCA normals), so the grid scores the identical readout the stage writes.

    uv run python scripts/score_contacts.py --predictions out/boulder_1/ out/boulder_3/
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

from tools.geometry.normals import load_scene_cloud  # noqa: E402
from tools.human_optim import (  # noqa: E402
    drift_gated_contact, joint_drift, joint_guarded_min_sdf,
    DETECT_MIN_FRAMES, merge_contact_spans,
    scene_sdf, vertex_to_joint, world_from_cam,
)
from tools.human_optim import paths  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.human_optim.stages import _ANN_TO_SMPLX, _hand_smplx_joints, resolve_annotation_path  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_DEFAULT_CORPUS_CONFIG = _REPO_ROOT / "configs" / "corpus.toml"
_DEFAULT_SMPLX_MODEL = _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"

# The 14-joint annotation (with its own ankle labels) splits foot(toe) from ankle and adds
# shoulders; legacy 10-joint annotations keep foot = ankle+foot union and have no ankles/shoulders.
_ANN_TO_SMPLX_14: dict[str, list[str]] = {
    "left_hand": _hand_smplx_joints("left"),
    "right_hand": _hand_smplx_joints("right"),
    "left_foot": ["left_foot"], "right_foot": ["right_foot"],
    "left_ankle": ["left_ankle"], "right_ankle": ["right_ankle"],
    "left_knee": ["left_knee"], "right_knee": ["right_knee"],
    "left_elbow": ["left_elbow"], "right_elbow": ["right_elbow"],
    "left_shoulder": ["left_shoulder"], "right_shoulder": ["right_shoulder"],
    "left_hip": ["left_hip"], "right_hip": ["right_hip"],
}
_EXTREMITIES_LEGACY = ("left_hand", "right_hand", "left_foot", "right_foot")
_EXTREMITIES_14 = _EXTREMITIES_LEGACY + ("left_ankle", "right_ankle")


def _ann_mapping(ann_names: list[str]) -> tuple[dict[str, list[str]], list[str], tuple[str, ...]]:
    """Pick the coarse→SMPLX mapping for one clip from its annotation joint names.

    Returns ``(mapping, coarse, extremities)`` — the 14-joint mapping when the annotation
    carries a separate ``left_ankle`` label, else the legacy 10-joint ``_ANN_TO_SMPLX``.
    ``extremities`` (= the FOCUS joints: hands + feet [+ ankles when 14]) drive the funnel's
    hover evidence and the pooled FOCUS row.
    """
    if "left_ankle" in ann_names:
        return _ANN_TO_SMPLX_14, list(_ANN_TO_SMPLX_14.keys()), _EXTREMITIES_14
    return _ANN_TO_SMPLX, list(_ANN_TO_SMPLX.keys()), _EXTREMITIES_LEGACY

# The readout grid. still_m = inf ⇒ no drift gate = the pure SDF baseline.
_CLOSE_GRID = (0.02, 0.03)
_STILL_GRID = (0.02, 0.025, 0.03, 0.05, float("inf"))
# A static SDF-only detector reference, printed alongside the pooled grid for comparison.
_SDF_DETECTOR_REF = "precision 0.83–0.97, recall 0.19–0.65, FPR 0.01–0.21 (hard speed veto)"


# ─────────────────────────────── scoring ─────────────────────────────────

def _members(mid_names: list[str], coarse: str, mapping: dict[str, list[str]]) -> list[int]:
    idx = {n: i for i, n in enumerate(mid_names)}
    return [idx[j] for j in mapping[coarse] if j in idx]


_ALLOW_FRAME_TRIM = False   # set by --allow-frame-trim; a frame-count mismatch is otherwise fatal


def _align_frames(n_body: int, n_ann: int, clip: str, oid: int) -> int:
    """Frames to score. A body/annotation frame-count mismatch is FATAL by default —
    blind min-trimming assumes the missing frames are at the tail, so any temporal
    offset would yield plausible-but-wrong scores. ``--allow-frame-trim`` restores
    the old warn+score-the-overlap behaviour."""
    if n_ann != n_body:
        msg = f"{clip} oid={oid}: annotation {n_ann}f vs body {n_body}f"
        if not _ALLOW_FRAME_TRIM:
            raise SystemExit(f"[frame-mismatch] {msg} — frame counts must match exactly "
                             f"(pass --allow-frame-trim to score the min-overlap anyway)")
        print(f"  [trim] {msg} → using {min(n_body, n_ann)}f")
    return min(n_body, n_ann)


def _coarse_pred(member_bool: np.ndarray) -> np.ndarray:
    """(N,) bool: a coarse joint is predicted in contact iff ANY member joint is."""
    return member_bool.any(axis=1) if member_bool.size else np.zeros(0, bool)


def _rates(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    f1_den = 2 * tp + fp + fn                                      # the user's "correlation" headline
    f1 = (2 * tp / f1_den) if f1_den else float("nan")            # 0 on all-miss; nan only if nothing scored
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, fpr=fpr, f1=f1,
               n_pos=tp + fn, n_neg=fp + tn)


def _score(pred_coarse: dict[str, np.ndarray], ann: np.ndarray, ann_names: list[str],
           valid: np.ndarray, coarse: list[str]) -> dict:
    """Confusion counts per coarse joint over frames where the annotation is 0/1 and valid.

    ``pred_coarse[c]`` is (N,) bool; ``ann`` is (J_ann, N) tri-state; returns per-joint
    and 'overall' dicts of tp/fp/fn/tn + precision/recall/fpr.
    """
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    out = {}
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for c in coarse:
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
    return (f"P={g(r['precision'])} R={g(r['recall'])} F1={g(r['f1'])} FPR={g(r['fpr'])}  "
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
                     fps: float, cc: dict, cg: dict, posed_world=None,
                     valid_mask=None):
    """Grid-invariant signals: posed geometry, guarded SDF, and motion drift.

    Everything the (close_thresh_m, still_m) sweep does NOT change — the pose, the scene
    SDF, the guarded per-joint min, and the net window drift — computed once. ``jw``/``vw``/``v2j``
    are returned for the span-merge distance guard and the confidence visibility witness.
    """
    J = mid.num_joints
    v_rest, j_rest = mid.forward_shape(betas[None])
    v2j = vertex_to_joint(v_rest[0], j_rest[0])                       # (V,) per person
    jw, vw = (_pose_world(mid, betas, q, scale)
              if posed_world is None else posed_world)
    sdf, dmin, conf = scene_sdf(vw, scene_pts, scene_normals, scene_conf,
                                prune_margin_m=float(cc["prune_margin_m"]))
    gsdf = joint_guarded_min_sdf(sdf, dmin, conf, v2j, J,
                                 max_dist_m=float(cc["max_dist_m"]), conf_thresh=float(cc["conf_thresh"]))
    drift = joint_drift(
        jw, fps=fps, window_s=float(cg["window_s"]), valid_mask=valid_mask)
    return jw, vw, v2j, gsdf, drift


def _load_scene_readout(pred: Path, cc: dict, cam_centers: torch.Tensor, device: str):
    """Scene cloud + PCA normals, gated exactly as ``estimate_contacts`` does."""
    sc = dict(np.load(pred / "geometry" / "scene.npz", allow_pickle=True))
    return load_scene_cloud(sc, cam_centers, conf_drop_q=float(cc.get("conf_drop_q", 0.0)),
                            k=int(cc["normal_k"]), device=device)


def _load_contact_cfg(path: Path) -> tuple[dict, dict]:
    """``([contacts], [contact_gate])`` with the same defaults as ``estimate_contacts``."""
    cfg = tomllib.load(path.open("rb")) if path.exists() else {}
    cc = cfg.get("contacts", {})
    cc.setdefault("max_dist_m", 0.10); cc.setdefault("conf_thresh", 0.1)
    cc.setdefault("conf_drop_q", 0.1); cc.setdefault("normal_k", 16)
    cc.setdefault("prune_margin_m", 0.10)
    cc.setdefault("close_thresh_m", 0.02)   # standard-readout enter distance (funnel reads it; grid sweeps it)
    cc.setdefault("exit_thresh_m", 0.035); cc.setdefault("min_span_s", 0.1)
    cc.setdefault("merge_dist_m", 0.03); cc.setdefault("merge_max_gap_s", 2.0)
    cc.setdefault("merge_sdf_m", 0.08)   # span-merge knobs
    cg = cfg.get("contact_gate", {})
    cg.setdefault("window_s", 0.35); cg.setdefault("still_m", 0.025)
    return cc, cg


def run_one(pred_dir: str, config_path: Path, device: str, source=None,
            body_path: Path | None = None, merge_dist: float = 0.0,
            gate_source: str = "final") -> list[dict]:
    """Score the readout grid on one clip's ``kinematic_1`` body (or ``body_path``).

    Returns a per-person list of ``{clip, oid, valid, grid}`` where ``grid`` maps
    ``(close_thresh_m, still_m) -> _score(...)`` (per-coarse-joint + overall confusion).
    ``merge_dist > 0`` applies the span-merge; ``gate_source='pre'`` feeds the stillness
    gate the persisted refine-start ``drift_pre`` instead of the final-body drift.
    """
    pred = Path(pred_dir)
    cc, cg = _load_contact_cfg(config_path)
    bpath = Path(body_path) if body_path is not None else paths.body_path(pred, "kinematic_1")
    kin = dict(np.load(bpath, allow_pickle=True))
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    if not bool(tr.get("metric", True)):  # absent flag = old already-metric artifact
        raise SystemExit(f"geometry under {pred} is up-to-scale — run estimate_scale after fuse")
    N, P = int(kin["num_frames"]), int(kin["num_people"])
    object_ids = np.asarray(kin["object_ids"], np.int32)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    cam_centers = world_from_cam(extr)[:, :3, 3]
    scene_pts, scene_normals, scene_conf = _load_scene_readout(pred, cc, cam_centers, device)

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    mid_names = list(mid.joint_names)
    exit_default = float(cc["exit_thresh_m"]); min_frames = DETECT_MIN_FRAMES

    ann_path = resolve_annotation_path(source or pred, None)
    ann_all = ann_names = ann_oids = None
    ann_ignored = coarse = focus = member_cols = None
    if ann_path.exists():
        a = dict(np.load(ann_path, allow_pickle=True))
        ann_all = np.asarray(a["contacts"])                       # (Pa, J_ann, N)
        ann_names = [str(x) for x in a["joint_names"]]
        ann_oids = [int(x) for x in np.asarray(a["object_ids"]).ravel()]
        ann_ignored = (np.asarray(a["ignored"]).ravel().astype(bool)
                       if "ignored" in a else np.zeros(len(ann_oids), bool))
        mapping, coarse, extremities = _ann_mapping(ann_names)    # per-clip 10- vs 14-joint
        focus = list(extremities)
        member_cols = {c: _members(mid_names, c, mapping) for c in coarse}

    results = []
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(kin["valid_mask"][p], bool)
        if ann_all is None or oid not in ann_oids:
            continue                                              # score only annotated people
        if bool(ann_ignored[ann_oids.index(oid)]):
            print(f"  [skip] {pred.name} oid={oid}: annotation flagged 'ignored'")
            continue
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        scale = float(kin["scale"][p]) if "scale" in kin else 1.0
        jw, _vw, _v2j, gsdf, drift = _readout_signals(
            mid, betas, q, scale, scene_pts, scene_normals,
            scene_conf, fps=fps, cc=cc, cg=cg, valid_mask=vmask)
        # gate source: read the persisted refine-start drift_pre (pre-enforcement) instead of
        # the final-body drift. Per-person fallback to computed drift if absent / NaN (old npz).
        gate_drift = drift
        if gate_source == "pre" and "drift_pre" in kin:
            dp = torch.as_tensor(
                np.asarray(kin["drift_pre"])[p], dtype=drift.dtype, device=drift.device)
            if dp.shape == drift.shape:
                gate_drift = torch.where(
                    torch.isfinite(dp) & torch.isfinite(drift), dp, drift)
        ann = ann_all[ann_oids.index(oid)]                        # (J_ann, N_ann)
        Nc = _align_frames(N, ann.shape[1], pred.name, oid)
        ann = ann[:, :Nc]; vmask_c = vmask[:Nc]
        grid = {}
        for close in _CLOSE_GRID:
            exit_below = max(exit_default, close)
            for still in _STILL_GRID:
                jc_t = drift_gated_contact(gsdf, gate_drift, close_thresh_m=close, exit_thresh_m=exit_below,
                                           min_frames=min_frames, still_m=still)
                if merge_dist > 0.0:
                    jc_t = merge_contact_spans(jc_t, jw, gsdf, merge_dist_m=merge_dist,
                                               max_gap_s=float(cc["merge_max_gap_s"]),
                                               gap_sdf_m=float(cc["merge_sdf_m"]), fps=fps)
                jc = jc_t.cpu().numpy()[:Nc]
                jc[~vmask_c] = False
                pred_coarse = {c: _coarse_pred(jc[:, member_cols[c]]) for c in coarse}
                grid[(close, still)] = _score(pred_coarse, ann, ann_names, vmask_c, coarse)
        results.append(dict(clip=pred.name, oid=oid, valid=int(vmask_c.sum()), n=N, grid=grid,
                            coarse=coarse, focus=focus))
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


def _pool_focus(all_res: list[dict], close: float, still: float) -> dict:
    """Pool confusion across people/clips over each clip's FOCUS joints (hands+feet[+ankles])."""
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for res in all_res:
        sc = res["grid"][(close, still)]
        for c in res["focus"]:
            if c in sc:
                for k in tot:
                    tot[k] += sc[c][k]
    return _rates(**tot)


def _print_pooled(all_res: list[dict], exclude: set[str]) -> None:
    pooled = [r for r in all_res if r["clip"] not in exclude]
    dropped = sorted({r["clip"] for r in all_res if r["clip"] in exclude})
    print("\n=== POOLED across clips (overall P / R / F1 / FPR per grid cell) ===")
    if dropped:
        print(f"  (excluded from the pool as fit outliers, shown per-clip above: {', '.join(dropped)})")
    print(f"  SDF-only detector reference: {_SDF_DETECTOR_REF}")
    for close in _CLOSE_GRID:
        print(f"  close_thresh={close * 100:.0f}cm")
        for still in _STILL_GRID:
            print(f"    still={_still_tag(still):>6s}  {_fmt(_pool(pooled, close, still))}")
    print("\n=== POOLED — FOCUS joints only (hands + feet + ankles; the signal joints) ===")
    for close in _CLOSE_GRID:
        print(f"  close_thresh={close * 100:.0f}cm")
        for still in _STILL_GRID:
            print(f"    still={_still_tag(still):>6s}  {_fmt(_pool_focus(pooled, close, still))}")


def _print_per_joint(all_res: list[dict], close: float, still: float) -> None:
    print(f"\n=== PER-COARSE-JOINT at close={close * 100:.0f}cm still={_still_tag(still)} ===")
    for res in all_res:
        print(f"  {res['clip']} oid={res['oid']}")
        sc = res["grid"][(close, still)]
        for c in res["coarse"]:
            if c in sc:
                print(f"    {c:12s} {_fmt(sc[c])}")


# ─────────────────────── the miss / FP funnel ────────────────
# For every annotated positive coarse-joint-frame the standard readout MISSES, walk the
# chain and charge the FIRST failing stage; split false positives into still-hover vs smear.
# The categories partition 1 − recall, so the table says which move buys what recall.
# Coarse joints are OR-folded, so each signal is judged by its BEST member: closest gsdf,
# stillest drift (optimistic — "even the best member fails here"). invisible / unlabeled are
# span-level (a hold never in the cloud / never pulled); the rest are per-frame.

_FUNNEL_LOOSE_GUARD_M = 0.15   # invisible test: no scene point within this on EVERY span frame = carve gap
_INF = 1e30                    # joint_guarded_min_sdf's +inf sentinel (finfo.max) reads as this in float32
_FUNNEL_CATS = ("invisible", "unlabeled", "hovering", "hovering_blind", "moving", "flicker")


@torch.no_grad()
def _funnel_signals(mid, betas, q, scale, scene_pts, scene_normals, scene_conf, *,
                    fps: float, cc: dict, cg: dict):
    """Per-person funnel signals ``(gsdf_tight, gsdf_loose, drift)`` (T,J) torch tensors.

    ``gsdf_tight`` uses the readout's ``[contacts] max_dist_m`` guard (the miss chain's
    distance test), ``gsdf_loose`` uses :data:`_FUNNEL_LOOSE_GUARD_M` (the visibility test),
    ``drift`` is the same net window drift the stillness gate consumes.
    """
    J = mid.num_joints
    v_rest, j_rest = mid.forward_shape(betas[None])
    v2j = vertex_to_joint(v_rest[0], j_rest[0])
    jw, vw = _pose_world(mid, betas, q, scale)
    sdf, dmin, conf = scene_sdf(vw, scene_pts, scene_normals, scene_conf,
                                prune_margin_m=float(cc["prune_margin_m"]))
    ct = float(cc["conf_thresh"])
    gsdf_tight = joint_guarded_min_sdf(sdf, dmin, conf, v2j, J, max_dist_m=float(cc["max_dist_m"]), conf_thresh=ct)
    gsdf_loose = joint_guarded_min_sdf(sdf, dmin, conf, v2j, J, max_dist_m=_FUNNEL_LOOSE_GUARD_M, conf_thresh=ct)
    drift = joint_drift(jw, fps=fps, window_s=float(cg["window_s"]))
    return gsdf_tight, gsdf_loose, drift


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D bool array → list of ``(start, end)`` half-open."""
    out, t, T = [], 0, len(mask)
    while t < T:
        if mask[t]:
            s = t
            while t < T and mask[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Dilate a 1-D bool mask by ±``k`` frames (annotation-adjacency for the smear split)."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    T = len(mask)
    for d in range(1, k + 1):
        out[np.clip(idx - d, 0, T - 1)] = True
        out[np.clip(idx + d, 0, T - 1)] = True
    return out


def _classify_person(gsdf_t, gsdf_l, drift, jc, auto_lab, ann, ann_names, valid,
                     member_cols, extremities, *, close: float, still_m: float, fps: float) -> dict:
    """Walk each missed annotated-positive coarse-joint-frame to its first-failing stage.

    Returns ``{per, tot, hover_gaps, still_ext_gsdf}``: per-coarse + overall confusion/category
    counts, the finite hovering gaps (m), and gsdf at still-extremity frames (m).
    """
    two_still = 2.0 * float(still_m)
    adj = max(1, int(round(0.2 * float(fps))))            # ±0.2 s = annotation-adjacent (smear)
    ann_idx = {n: i for i, n in enumerate(ann_names)}
    per, hover_gaps, still_ext = {}, [], []
    for c, M in member_cols.items():
        if c not in ann_idx or not M:
            continue
        a = ann[ann_idx[c]]                                # (N,) tri-state
        gt_all = gsdf_t[:, M]                               # (N, |M|) tight guard, per member
        jstar = gt_all.argmin(axis=1)                      # best (closest-to-wall) member per frame
        rows = np.arange(gt_all.shape[0])
        gt = gt_all[rows, jstar]                            # (N,) closest member's tight gsdf
        dr = drift[:, M][rows, jstar]                       # (N,) THAT member's drift (not a stiller sibling's)
        gl = gsdf_l[:, M].min(axis=1)                       # (N,) any member visible? (loose guard)
        lab = (auto_lab[:, M] == 1).any(axis=1)            # (N,) any member frozen-labeled
        pred = jc[:, M].any(axis=1)                        # (N,) standard readout OR-fold
        pos = valid & (a == 1)
        neg = valid & (a == 0)
        cj = {"pos": int(pos.sum()), "hit": int((pos & pred).sum()),
              "neg": int(neg.sum()), "fp": int((neg & pred).sum())}
        cj.update({k: 0 for k in _FUNNEL_CATS})
        if c in extremities:                               # hover evidence: still + visible extremity frames
            sel = valid & (dr < two_still) & (gt < _INF)
            still_ext.extend(gt[sel].tolist())
        fp = neg & pred
        adj_mask = _dilate(pos, adj)
        cj["smear"] = int((fp & adj_mask).sum())
        cj["still_hover"] = int((fp & ~adj_mask).sum())
        for s, e in _spans(pos):
            span_invis = bool((gl[s:e] >= _INF).all())     # never any scene point near the joint, whole span
            span_unlab = bool(not lab[s:e].any())          # the in-loop detector never fired here
            for t in range(s, e):
                if pred[t]:
                    continue                               # a hit, not a miss
                if span_invis:
                    cat = "invisible"
                elif span_unlab:
                    cat = "unlabeled"
                elif gt[t] >= close:
                    cat = "hovering_blind" if gt[t] >= _INF else "hovering"
                    if cat == "hovering":
                        hover_gaps.append(float(gt[t]))
                elif dr[t] >= two_still:
                    cat = "moving"
                else:
                    cat = "flicker"
                cj[cat] += 1
        per[c] = cj
    tot = _merge(list(per.values()))
    return dict(per=per, tot=tot, hover_gaps=hover_gaps, still_ext_gsdf=still_ext)


def _funnel_one(pred_dir: str, config_path: Path, device: str, source=None,
                body_path: Path | None = None) -> list[dict]:
    """Run the funnel on one clip's ``kinematic_1`` body (or ``body_path``) → per-person dicts."""
    pred = Path(pred_dir)
    cc, cg = _load_contact_cfg(config_path)
    close = float(cc["close_thresh_m"]); still_m = float(cg["still_m"])
    exit_below = max(float(cc["exit_thresh_m"]), close)
    bpath = Path(body_path) if body_path is not None else paths.body_path(pred, "kinematic_1")
    kin = dict(np.load(bpath, allow_pickle=True))
    if "auto_labels" not in kin:
        raise SystemExit(f"{bpath} has no 'auto_labels' — re-run kinematics_1 with auto-contact labels first")
    tr = dict(np.load(pred / "geometry" / "transform.npz", allow_pickle=True))
    if not bool(tr.get("metric", True)):  # absent flag = old already-metric artifact
        raise SystemExit(f"geometry under {pred} is up-to-scale — run estimate_scale after fuse")
    N, P = int(kin["num_frames"]), int(kin["num_people"])
    object_ids = np.asarray(kin["object_ids"], np.int32)
    fps = float(tr.get("fps", kin.get("fps", 30.0)))
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    cam_centers = world_from_cam(extr)[:, :3, 3]
    scene_pts, scene_normals, scene_conf = _load_scene_readout(pred, cc, cam_centers, device)

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)

    ann_path = resolve_annotation_path(source or pred, None)
    if not ann_path.exists():
        return []
    a = dict(np.load(ann_path, allow_pickle=True))
    ann_all = np.asarray(a["contacts"])
    ann_names = [str(x) for x in a["joint_names"]]
    ann_oids = [int(x) for x in np.asarray(a["object_ids"]).ravel()]
    ann_ignored = (np.asarray(a["ignored"]).ravel().astype(bool)
                   if "ignored" in a else np.zeros(len(ann_oids), bool))
    mapping, _coarse, extremities = _ann_mapping(ann_names)       # per-clip 10- vs 14-joint
    member_cols = {c: _members(list(mid.joint_names), c, mapping) for c in _coarse}

    results = []
    for p in range(P):
        oid = int(object_ids[p])
        vmask = np.asarray(kin["valid_mask"][p], bool)
        if oid not in ann_oids:
            continue
        if bool(ann_ignored[ann_oids.index(oid)]):
            print(f"  [skip] {pred.name} oid={oid}: annotation flagged 'ignored'")
            continue
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        scale = float(kin["scale"][p]) if "scale" in kin else 1.0
        gt_t, gl_t, dr_t = _funnel_signals(mid, betas, q, scale, scene_pts, scene_normals,
                                           scene_conf, fps=fps, cc=cc, cg=cg)
        jc = drift_gated_contact(gt_t, dr_t, close_thresh_m=close, exit_thresh_m=exit_below,
                                 min_frames=DETECT_MIN_FRAMES, still_m=still_m).cpu().numpy()
        jc[~vmask] = False
        auto_lab = np.asarray(kin["auto_labels"][p], np.int8)
        ann = ann_all[ann_oids.index(oid)]
        Nc = _align_frames(N, ann.shape[1], pred.name, oid)
        cls = _classify_person(gt_t.cpu().numpy()[:Nc], gl_t.cpu().numpy()[:Nc], dr_t.cpu().numpy()[:Nc],
                               jc[:Nc], auto_lab[:Nc], ann[:, :Nc], ann_names, vmask[:Nc], member_cols,
                               extremities, close=close, still_m=still_m, fps=fps)
        results.append(dict(clip=pred.name, oid=oid, valid=int(vmask[:Nc].sum()), n=N,
                            extremities=list(extremities), **cls))
    return results


def _merge(dicts: list[dict]) -> dict:
    """Sum the integer entries of several confusion/category dicts key-wise."""
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, np.integer)):
                out[k] = out.get(k, 0) + int(v)
    return out


def _frac(n: int, d: int) -> float:
    return (n / d) if d else float("nan")


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else float("nan")


def _stat(xs: list) -> str:
    if not xs:
        return "n=0"
    a = np.asarray(xs)
    return (f"median {np.median(a) * 100:.1f}cm "
            f"[p25 {np.percentile(a, 25) * 100:.1f}, p75 {np.percentile(a, 75) * 100:.1f}] n={len(a)}")


def _miss_line(t: dict) -> str:
    def pc(k):
        return f"{_pct(t.get(k, 0), t['pos']):.0f}%"
    hb = f" (blind {pc('hovering_blind')})" if t.get("hovering_blind", 0) else ""
    return (f"invisible {pc('invisible')}  unlabeled {pc('unlabeled')}  "
            f"hovering {pc('hovering')}{hb}  moving {pc('moving')}  flicker {pc('flicker')}")


def _print_funnel_block(t: dict, gaps: list, sext: list) -> None:
    miss = t["pos"] - t["hit"]
    print(f"  positives {t['pos']}  recall {_frac(t['hit'], t['pos']):.3f}  (hit {t['hit']}, miss {miss})")
    print(f"  miss causes (% of positives): {_miss_line(t)}")
    print(f"  hovering gap:            {_stat(gaps)}")
    print(f"  still-extremity gsdf:    {_stat(sext)}")
    print(f"  FP: neg {t['neg']}  fp {t['fp']}  fpr {_frac(t['fp'], t['neg']):.3f}  "
          f"= still-hover {_pct(t.get('still_hover', 0), t['fp']):.0f}%  smear {_pct(t.get('smear', 0), t['fp']):.0f}%")


def _print_funnel(all_res: list[dict], exclude: set[str]) -> None:
    for res in all_res:
        print(f"\n### FUNNEL — {res['clip']} oid={res['oid']} ({res['valid']}/{res['n']} valid) ###")
        _print_funnel_block(res["tot"], res["hover_gaps"], res["still_ext_gsdf"])
    pooled = [r for r in all_res if r["clip"] not in exclude]
    dropped = sorted({r["clip"] for r in all_res if r["clip"] in exclude})
    tot = _merge([r["tot"] for r in pooled])
    gaps = [g for r in pooled for g in r["hover_gaps"]]
    sext = [g for r in pooled for g in r["still_ext_gsdf"]]
    print(f"\n### FUNNEL POOLED{' (excl ' + ', '.join(dropped) + ')' if dropped else ''} ###")
    _print_funnel_block(tot, gaps, sext)
    print("  per-coarse (pooled extremities):")
    ext_order = list(dict.fromkeys(c for r in pooled for c in r["extremities"]))   # ordered union
    for c in ext_order:
        cjs = [r["per"][c] for r in pooled if c in r["per"]]
        if not cjs:
            continue
        cj = _merge(cjs)
        print(f"    {c:11s} pos {cj['pos']:4d}  R {_frac(cj['hit'], cj['pos']):.2f}  | {_miss_line(cj)}")


def _config_path(value: str | Path) -> Path:
    """Resolve a corpus-config path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Score selected prediction targets using behavior from ``corpus.toml``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", nargs="+", required=True, help="one or more clip dirs")
    ap.add_argument("--source", default=None, help="annotation override (default: data/<stem>.npz per clip)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CORPUS_CONFIG)
    ap.add_argument("--body-path", default=None,
                    help="score this body npz instead of kinematic_1.npz (requires a single --predictions)")
    ap.add_argument("--funnel", action="store_true",
                    help="run the miss/FP funnel (needs auto_labels in the body npz) instead of the grid")
    args = ap.parse_args(argv)
    try:
        with args.config.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
        corpus_config = raw_config.get("corpus", {})
        score_config = raw_config.get("score", {})
        contact_config = _config_path(
            corpus_config.get("human_optim_config", _DEFAULT_CONFIG)
        )
        device = str(score_config.get("device", "cuda"))
        ref_close = float(score_config.get("ref_close", 0.02))
        ref_still = float(score_config.get("ref_still", 0.025))
        exclude_from_pool = {
            str(value) for value in score_config.get("exclude_from_pool", ["sport_climb"])
        }
        merge_value = score_config.get("merge_dist", "")
        merge_dist = None if merge_value == "" else float(merge_value)
        gate_value = str(score_config.get("gate_source", ""))
        gate_source = gate_value or None
        if gate_source not in {None, "final", "pre"}:
            raise ValueError("[score].gate_source must be empty, 'final', or 'pre'")
        allow_frame_trim = bool(score_config.get("allow_frame_trim", False))
    except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        ap.error(f"invalid corpus config: {error}")
    global _ALLOW_FRAME_TRIM
    _ALLOW_FRAME_TRIM = allow_frame_trim
    body_path = Path(args.body_path) if args.body_path else None
    if body_path is not None and len(args.predictions) != 1:
        raise SystemExit("--body-path requires exactly one --predictions")

    if args.funnel:
        all_res: list[dict] = []
        for pd in args.predictions:
            all_res.extend(_funnel_one(
                pd, contact_config, device, source=args.source, body_path=body_path
            ))
        if not all_res:
            print("no annotated people found in the given clips")
            return 1
        _print_funnel(all_res, exclude_from_pool)
        return 0

    grid_merge = 0.0 if merge_dist is None else merge_dist
    grid_gate = gate_source or "final"
    all_res = []
    for pd in args.predictions:
        all_res.extend(run_one(pd, contact_config, device, source=args.source, body_path=body_path,
                               merge_dist=grid_merge, gate_source=grid_gate))
    if not all_res:
        print("no annotated people found in the given clips")
        return 1
    if grid_merge > 0.0 or grid_gate != "final":
        print(f"[readout: merge_dist={grid_merge} gate_source={grid_gate}]")
    for res in all_res:
        _print_clip_grid(res)
    _print_per_joint(all_res, ref_close, ref_still)
    _print_pooled(all_res, exclude_from_pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
