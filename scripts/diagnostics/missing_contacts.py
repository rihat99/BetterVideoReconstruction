"""Diagnose missing contacts from the dynamics root-wrench residual.

The dynamics stage solves contact forces at 6 groups (hands, feet, ankles) so
the RNEA base wrench — the un-actuated 6-DOF residual force/torque on the
free-flyer root — vanishes. When the auto contact labels *miss* a real contact,
the solver has nowhere to put the needed force and the residual stays high. This
diagnostic asks, per frame and per limb: **how much of the residual could a
contact at this limb explain?** — and scores that against the manual GT
annotations in ``data/<stem>.npz``.

Two scores, both residual-driven:

* ``probe``   — closed-form least squares: the best single world force ``f`` at
  the limb position ``p`` changes the root wrench by ``[f ; (p − p_root) × f]``;
  the score is how much of ``‖w‖²`` that can cancel (absolute, and relative).
  Each limb is probed alone, so this is an upper bound per limb.
* ``resolve`` — the honest counterfactual: re-run the stage's force solver with
  every group unmasked on every valid frame, so limbs compete under the real
  smoothness/magnitude priors. The score is the solved force magnitude on
  frames where the auto labels said "no contact".

Baselines for comparison: negative distance-to-scene (``joint_distance_cm``)
and negative drift (``drift_pre``) — the ingredients auto detection already
uses. Evaluation is at 4-limb level (foot + ankle fold per side: the annotation
cannot distinguish heel from toe), AUROC/AP over (person, frame, limb)
candidates where the auto label is off, positives = GT contact, GT −1 ignored.

Reads only finished ``out/<stem>/`` trees + ``data/<stem>.npz``; writes
per-scene scores to ``scratch/missing_contacts/`` and prints a summary table.
Main-pipeline code is reused read-only (nothing is modified or overwritten).

Usage::

    uv run python scripts/diagnostics/missing_contacts.py            # all scenes
    uv run python scripts/diagnostics/missing_contacts.py --scenes boulder_1
    uv run python scripts/diagnostics/missing_contacts.py --no-resolve   # probe only
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import dataclasses  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.human_optim.config import load_config  # noqa: E402
from tools.human_optim.dynamics import _GRAVITY_MAG, _contact_catalog, _solve_person  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_OUT_DIR = _REPO_ROOT / "scratch" / "missing_contacts"

# GT defects found by inspection: competition_2 frames 68-178 claim "no contact"
# for every person and limb for 4.4 s while the video shows the climber mid-wall
# (annotation left in the tool's default state). Scored as unknown (-1).
_GT_DEFECTS: dict[str, tuple[int, int]] = {"competition_2": (68, 179)}

# Evaluation limbs: annotation name -> the dynamics contact groups it folds over.
_LIMBS = {
    "left_hand": ["left_hand"],
    "right_hand": ["right_hand"],
    "left_foot": ["left_foot", "left_ankle"],
    "right_foot": ["right_foot", "right_ankle"],
}


def _skew(r: np.ndarray) -> np.ndarray:
    """Cross-product matrix: ``_skew(r) @ f == np.cross(r, f)``. r: (..., 3) -> (..., 3, 3)."""
    K = np.zeros(r.shape[:-1] + (3, 3), r.dtype)
    K[..., 0, 1], K[..., 0, 2] = -r[..., 2], r[..., 1]
    K[..., 1, 0], K[..., 1, 2] = r[..., 2], -r[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -r[..., 1], r[..., 0]
    return K


def probe_reduction(w_world: np.ndarray, lever: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-case residual reduction by ONE force per candidate contact point.

    ``w_world``: (N, 6) root wrench [force | torque about the root, world frame].
    ``lever``: (N, C, 3) contact point minus root position. Returns
    ``(reduction (N, C), relative (N, C))`` with ``reduction = ‖w‖² − min_f ‖w − A f‖²``,
    ``A = [I ; skew(lever)]`` — closed form via the 3×3 normal equations.
    """
    K = _skew(lever)                                       # (N, C, 3, 3)
    M = np.eye(3, dtype=w_world.dtype) + np.einsum("ncji,ncjk->ncik", K, K)
    b = w_world[:, None, :3] - np.einsum("ncij,nj->nci", K, w_world[:, 3:])  # Aᵀw
    f = np.linalg.solve(M, b[..., None])[..., 0]           # (N, C, 3)
    reduction = np.einsum("nci,nci->nc", b, f)             # LS identity: Δ = (Aᵀw)ᵀ f*
    w2 = np.einsum("ni,ni->n", w_world, w_world)
    return reduction, reduction / np.maximum(w2, 1e-12)[:, None]


def probe_torque(w_world: np.ndarray, lever: np.ndarray) -> np.ndarray:
    """Torque-residual reduction when the limb carries exactly the missing force.

    Any limb can cancel the force part of the residual alone, so the free-force
    probe barely separates limbs. This variant pins the force to ``w_f`` (the
    unexplained force is what it is) and asks *where* it must act for the torque
    residual to shrink too: score ``‖w_τ‖² − ‖w_τ − lever × w_f‖²`` per (N, C).
    Negative = a contact at this limb would make the torque books worse.
    """
    tq_after = w_world[:, None, 3:] - np.cross(lever, w_world[:, None, :3])
    return (np.einsum("ni,ni->n", w_world[:, 3:], w_world[:, 3:])[:, None]
            - np.einsum("nci,nci->nc", tq_after, tq_after))


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Root quaternion (N, 4) -> rotation matrices (N, 3, 3) via better_robot."""
    from better_robot.lie import so3
    return so3.to_matrix(torch.as_tensor(quat)).numpy()


class PersonDynamics:
    """One person's frozen-q dynamics context: model, world wrenches, lever arms.

    Recomputes the *no-contact* base wrench with the same model/gravity the stage
    used, rotates all wrenches to the world frame, and self-checks that the saved
    residual equals the no-contact wrench minus each saved force's ``[f ; r × f]``
    contribution — which validates frame, ordering and sign conventions in one shot.
    """

    def __init__(self, mid, betas: np.ndarray, q: np.ndarray, valid: np.ndarray,
                 gravity_world: np.ndarray, fps: float, device: str,
                 smooth_sigma: float = 0.0) -> None:
        from better_robot.dynamics import rnea
        from tools.smplx_robot.dynamics import velocity_acceleration_from_trajectory

        dev = torch.device(device)
        model = mid.shaped(torch.as_tensor(betas, dtype=torch.float32, device=dev)).robot
        g = torch.as_tensor(gravity_world, dtype=torch.float32, device=dev)
        g = g / (g.norm() + 1e-9) * _GRAVITY_MAG
        g6 = torch.cat([g, torch.zeros(3, device=dev)])
        self.model = dataclasses.replace(
            model, values=dataclasses.replace(model.values, gravity=g6))
        self.q_model = normalize_q(torch.as_tensor(q, dtype=torch.float32, device=dev),
                                   mid.num_joints)
        with torch.no_grad():
            self.v, self.a = velocity_acceleration_from_trajectory(
                self.model, self.q_model, 1.0 / fps)
            if smooth_sigma > 0:                       # temporal Gaussian on v, a (frame axis)
                from scipy.ndimage import gaussian_filter1d
                self.v = torch.as_tensor(gaussian_filter1d(
                    self.v.cpu().numpy(), smooth_sigma, axis=0), device=dev)
                self.a = torch.as_tensor(gaussian_filter1d(
                    self.a.cpu().numpy(), smooth_sigma, axis=0), device=dev)
            wnc = rnea(self.model, self.q_model, self.v, self.a)[..., :6].cpu().numpy()
        self.R_root = _quat_to_matrix(self.q_model[:, 3:7].cpu().numpy())     # (N, 3, 3)
        self.wnc_world = self.rotate(wnc)
        self.valid = np.asarray(valid, bool)

    def rotate(self, w_local: np.ndarray) -> np.ndarray:
        """Root-local wrench (N, 6) -> world frame (still about the root point)."""
        return np.concatenate([
            np.einsum("nij,nj->ni", self.R_root, w_local[:, :3]),
            np.einsum("nij,nj->ni", self.R_root, w_local[:, 3:])], axis=1)

    def check(self, w_res_local: np.ndarray, forces_world: np.ndarray,
              lever: np.ndarray) -> float:
        """Max |w_nc − Σ[f ; r × f] − w_res| over valid frames (world, mixed units)."""
        contrib = np.concatenate([
            forces_world.sum(1),
            np.cross(lever, forces_world).sum(1)], axis=1)                    # (N, 6)
        err = self.wnc_world - contrib - self.rotate(w_res_local)
        return float(np.abs(err[self.valid]).max()) if self.valid.any() else 0.0


def load_scene(stem: str, cfg, device: str, pred: Path | None = None,
               ann_path: Path | None = None) -> dict:
    """Load one scene's saved stage outputs + GT annotation, aligned by object id.

    ``pred`` overrides the default ``out/<stem>/`` tree (e.g. a DB symlink
    farm); ``ann_path`` overrides ``data/<stem>.npz``. A missing
    ``dynamics.npz`` is tolerated — the residual/force/position fields are then
    ``None`` (the completion solve recomputes what it needs).
    """
    pred = Path(pred) if pred is not None else _REPO_ROOT / "out" / stem
    kin = dict(np.load(pred / "human_optim" / "kinematic_1.npz", allow_pickle=True))
    cnt = dict(np.load(pred / "human_optim" / "contacts_1.npz", allow_pickle=True))
    dyn_path = pred / "human_optim" / "dynamics.npz"
    dyn = dict(np.load(dyn_path, allow_pickle=True)) if dyn_path.exists() else None
    ann = dict(np.load(ann_path or _REPO_ROOT / "data" / f"{stem}.npz", allow_pickle=True))
    joint_names = [str(n) for n in kin["joint_names"]]
    ann_names = [str(n) for n in ann["joint_names"]]
    ann_oids = list(np.asarray(ann["object_ids"], np.int32))
    cnt_oids = list(np.asarray(cnt["object_ids"], np.int32))

    groups = list(cfg.knobs["contact"]["groups"])
    catalog = _contact_catalog(joint_names)
    members = {g: [joint_names.index(m) for m in catalog[g][1]] for g in groups}

    N = int(kin["num_frames"])
    people = []
    for p, oid in enumerate(np.asarray(kin["object_ids"], np.int32)):
        if int(oid) not in ann_oids:
            print(f"  [warn] {stem}: object {oid} has no GT annotation row — skipped")
            continue
        a = ann_oids.index(int(oid))
        cp = cnt_oids.index(int(oid)) if int(oid) in cnt_oids else p

        def gt_row(limb: str) -> np.ndarray:
            # 14-joint annotations split heel/toe: fold the ankle row into the foot
            # limb (contact if either; unknown only when both are unknown).
            rows = [ann["contacts"][a, ann_names.index(limb)]]
            ankle = limb.replace("_foot", "_ankle")
            if limb.endswith("_foot") and ankle in ann_names:
                rows.append(ann["contacts"][a, ann_names.index(ankle)])
            r = np.stack(rows)
            return np.where((r == 1).any(0), 1, np.where((r == -1).all(0), -1, 0))

        gt = np.stack([gt_row(k) for k in _LIMBS], 1)                               # (N, 4)
        if stem in _GT_DEFECTS:
            lo, hi = _GT_DEFECTS[stem]
            gt[lo:hi] = -1
        auto52 = np.asarray(cnt["joint_contact"][cp], bool)                         # (N, 52)
        group_auto = np.stack([auto52[:, members[g]].any(1) for g in groups], 1)    # (N, 6)
        dist52 = np.asarray(cnt["joint_distance_cm"][cp], np.float32)
        drift52 = np.asarray(kin["drift_pre"][p], np.float32)
        people.append(dict(
            oid=int(oid), betas=kin["betas"][p], q=kin["q"][p],
            valid=np.asarray(kin["valid_mask"][p], bool),
            gt=gt[:N], auto52=auto52, group_auto=group_auto,
            w_res_local=(np.asarray(dyn["base_wrench"][p], np.float32)
                         if dyn is not None else None),
            forces_world=(np.asarray(dyn["contact_forces"][p], np.float32)
                          if dyn is not None else None),                            # (N, 6, 3)
            root_pos=(np.asarray(dyn["joints_world"][p, :, 0], np.float32)
                      if dyn is not None else None),
            group_pos=(np.stack([dyn["joints_world"][p, :, joint_names.index(catalog[g][0])]
                                 for g in groups], 1)
                       if dyn is not None else None),                               # (N, 6, 3)
            # joint_distance_cm is NaN for joints with no nearby scene point; treat those
            # members as "far" so one pruned finger cannot poison the whole hand group.
            group_dist=np.stack([np.where(np.isnan(dist52[:, members[g]]), np.inf,
                                          dist52[:, members[g]]).min(1) for g in groups], 1),
            group_drift=np.stack([drift52[:, members[g]].min(1) for g in groups], 1),
        ))

    grav_path = pred / "geocalib" / "gravity.npz"
    gravity = (np.asarray(np.load(grav_path)["gravity_world"], np.float32)
               if grav_path.exists() else np.array([0.0, 1.0, 0.0], np.float32))
    return dict(stem=stem, pred=pred, people=people, groups=groups, members=members,
                joint_names=joint_names, fps=float(kin["fps"]), gravity=gravity, N=N)


def counterfactual_solve(scene: dict, person: dict, mid, cfg, device: str) -> np.ndarray:
    """Re-run the stage's force solver with every group unmasked on valid frames.

    Returns the solved world force per group ``(N, 6, 3)`` — where the solver
    *wants* force once it is allowed everywhere.
    """
    joint_names = scene["joint_names"]
    contact_all = person["auto52"].copy()
    for g in scene["groups"]:
        rep = _contact_catalog(joint_names)[g][0]
        contact_all[:, joint_names.index(rep)] = True

    friction = cfg.knobs.get("friction")
    scene_pts = scene_normals = None
    sc_path, tr_path = scene["pred"] / "geometry" / "scene.npz", scene["pred"] / "geometry" / "transform.npz"
    if friction is not None and sc_path.exists() and tr_path.exists():
        from tools.geometry.normals import load_scene_cloud
        from tools.human_optim.camera import world_from_cam
        tr = dict(np.load(tr_path, allow_pickle=True))
        cam_centers = world_from_cam(
            torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device))[:, :3, 3]
        scene_pts, scene_normals, _ = load_scene_cloud(
            dict(np.load(sc_path, allow_pickle=True)), cam_centers,
            conf_drop_q=0.0, k=int(friction["normal_k"]), device=device)

    res = _solve_person(
        mid, torch.as_tensor(person["betas"], dtype=torch.float32),
        torch.as_tensor(person["q"], dtype=torch.float32),
        torch.as_tensor(contact_all, dtype=torch.bool, device=device),
        person["valid"], scene["gravity"], weights=cfg.weights_for("dynamics"),
        groups=scene["groups"], dyn_opt=cfg.knobs["dynamics_optimizer"],
        fps=scene["fps"], device=device, label=f"obj {person['oid']} unmasked",
        scene_pts=scene_pts, scene_normals=scene_normals, friction=friction)
    return np.asarray(res["contact_forces"], np.float32)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney); nan when one class is empty."""
    from scipy.stats import rankdata
    npos = int(labels.sum())
    if npos == 0 or npos == len(labels):
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * (len(labels) - npos)))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hit = labels[order]
    precision = np.cumsum(hit) / (np.arange(len(hit)) + 1)
    return float((precision * hit).sum() / hit.sum())


def fold_limbs(group_scores: np.ndarray, groups: list[str], reduce=np.max) -> np.ndarray:
    """(N, 6) per-group scores -> (N, 4) per-limb via max (foot = max(foot, ankle))."""
    idx = {g: i for i, g in enumerate(groups)}
    return np.stack([reduce(group_scores[:, [idx[g] for g in gs]], axis=1)
                     for gs in _LIMBS.values()], 1)


def evaluate_scene(scene: dict, score_maps: dict[str, list[np.ndarray]]) -> dict:
    """AUROC/AP per score over all (person, frame, limb) candidates with auto off."""
    cand_scores = {k: [] for k in score_maps}
    cand_labels = []
    for p, person in enumerate(scene["people"]):
        limb_auto = fold_limbs(person["group_auto"].astype(float), scene["groups"]) > 0
        mask = person["valid"][:, None] & ~limb_auto & (person["gt"] >= 0)      # (N, 4)
        cand_labels.append((person["gt"] == 1)[mask])
        for k, per_person in score_maps.items():
            cand_scores[k].append(per_person[p][mask])
    labels = np.concatenate(cand_labels)
    out = dict(n=len(labels), pos=int(labels.sum()))
    for k in score_maps:
        s = np.concatenate(cand_scores[k])
        out[k] = (auroc(s, labels), average_precision(s, labels))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default=None, help="comma-separated stems (default: all data/ "
                                                   "scenes except sport_climb)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resolve", action="store_true", help="skip the counterfactual re-solve")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.scenes:
        stems = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        stems = sorted(p.stem for p in (_REPO_ROOT / "data").glob("*.npz")
                       if p.stem != "sport_climb")

    from tools.human_optim.stages import _DEFAULT_SMPLX_MODEL
    from tools.smplx_robot import load_smplx_mid
    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, args.device)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in stems:
        scene = load_scene(stem, cfg, args.device)
        if not scene["people"]:
            continue
        maps: dict[str, list[np.ndarray]] = {k: [] for k in
                                             ("probe_abs", "probe_rel", "probe_tq",
                                              "wnorm", "dist", "drift")}
        if not args.no_resolve:
            maps["resolve"] = []
        save: dict[str, list[np.ndarray]] = {}
        recon_err = 0.0
        for person in scene["people"]:
            pd = PersonDynamics(mid, person["betas"], person["q"], person["valid"],
                                scene["gravity"], scene["fps"], args.device)
            lever = person["group_pos"] - person["root_pos"][:, None]           # (N, 6, 3)
            recon_err = max(recon_err, pd.check(person["w_res_local"],
                                                person["forces_world"], lever))
            w_world = pd.rotate(person["w_res_local"])
            red, rel = probe_reduction(w_world, lever)
            tq = probe_torque(w_world, lever)
            maps["probe_abs"].append(fold_limbs(red, scene["groups"]))
            maps["probe_rel"].append(fold_limbs(rel, scene["groups"]))
            maps["probe_tq"].append(fold_limbs(tq, scene["groups"]))
            wn = np.linalg.norm(w_world, axis=1, keepdims=True)                 # limb-blind
            maps["wnorm"].append(fold_limbs(np.repeat(wn, 6, 1), scene["groups"]))
            maps["dist"].append(fold_limbs(-person["group_dist"], scene["groups"]))
            maps["drift"].append(fold_limbs(-person["group_drift"], scene["groups"]))
            if not args.no_resolve:
                f_all = counterfactual_solve(scene, person, mid, cfg, args.device)
                maps["resolve"].append(fold_limbs(np.linalg.norm(f_all, axis=-1),
                                                  scene["groups"]))
            for key, arr in [("probe_abs6", red), ("probe_rel6", rel), ("probe_tq6", tq),
                             ("w_world", w_world), ("wnc_world", pd.wnc_world)]:
                save.setdefault(key, []).append(arr)
            if not args.no_resolve:
                save.setdefault("resolve_f6", []).append(f_all)

        row = evaluate_scene(scene, maps)
        row["stem"], row["recon_err"] = stem, recon_err
        rows.append(row)
        np.savez(_OUT_DIR / f"{stem}.npz",
                 object_ids=np.array([q["oid"] for q in scene["people"]], np.int32),
                 groups=np.array(scene["groups"]), limbs=np.array(list(_LIMBS)),
                 gt=np.stack([q["gt"] for q in scene["people"]]),
                 group_auto=np.stack([q["group_auto"] for q in scene["people"]]),
                 valid=np.stack([q["valid"] for q in scene["people"]]),
                 **{k: np.stack(v) for k, v in save.items()},
                 **{k: np.stack(v) for k, v in maps.items()})
        parts = "  ".join(f"{k} {row[k][0]:.3f}/{row[k][1]:.3f}" for k in maps)
        print(f"{stem:22s} cand={row['n']:5d} pos={row['pos']:5d} "
              f"recon_err={recon_err:8.2f}  AUROC/AP: {parts}", flush=True)

    print("\nmacro means over scenes (AUROC | AP):")
    for k in rows[0]:
        if isinstance(rows[0].get(k), tuple):
            a = np.array([r[k][0] for r in rows if not np.isnan(r[k][0])])
            p = np.array([r[k][1] for r in rows if not np.isnan(r[k][1])])
            print(f"  {k:10s} {a.mean():.3f} | {p.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
