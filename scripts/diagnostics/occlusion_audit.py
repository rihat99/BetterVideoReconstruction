"""Audit the contact stage's body-occlusion computation against an exact ray cast.

The contact stage (``scripts/stages/estimate_contacts.py``) weights each per-joint
contact label by a 3-D visibility term built from
:func:`tools.human_optim.contacts.joint_occlusion_counts`. That routine counts how
many of ``2500`` *sampled* body vertices lie inside a 2 cm tube around the
camera-to-joint ray and slightly in front of the joint's own front skin; five such
vertices zero the joint's visibility.

This diagnostic re-poses the saved bodies exactly as the stage does and, per
``(frame, joint)``, casts one or more rays from the camera centre through the joint
and intersects them (Moller-Trumbore, batched on the GPU) against the
full-resolution triangles of every tracked person. A triangle is the target joint's
*own* skin when **any** of its three vertices is owned by a joint in the target's
occlusion group (conservative, avoids seam holes). Per ray we record the nearest
own-group hit depth and the nearest non-own hit depth, which yields two occlusion
criteria that are scored against the current method:

* **joint-centre** — ``gap = joint_depth − nearest_non_own_depth`` (``−inf`` with no
  non-own hit); occluded at ``tau`` iff ``gap > tau`` (tau ∈ {3,5,8} cm).
* **own-surface reference** — occluded iff a non-own hit lies more than a small
  ``front`` margin in front of the joint's own visible skin (nearest own-group hit;
  falls back to the joint depth when the ray misses all own skin — a silhouette
  case whose rate is recorded).

Also computed: a 5-ray (centre + 4 perpendicular ~1.5 cm offsets) robustness check
that measures how often the single centre ray disagrees with the multi-ray fraction,
and single- vs multi-ray temporal flicker on stable-pose joint-frames.

Outputs (under ``--save-dir``): a JSON of confusion tables / per-joint rates /
disagreement examples / timing, a per-scene ``.npz`` of the raw counts+gaps, and a
few overlay PNGs of the most confident disagreements.

Run::

    CUDA_VISIBLE_DEVICES=3 uv run --no-sync python scripts/diagnostics/occlusion_audit.py \
        --predictions out --scenes sport_climb boulder_1 gtIGgs_y2nY_0024 \
        --save-dir scratch/occlusion_audit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.human_optim import (  # noqa: E402
    joint_occlusion_counts, project, transform_points, vertex_to_joint,
    world_from_cam,
)
from tools.human_optim.confidence import occlusion_group_ids  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

# Joints the audit reports individually (the hypotheses are about these).
REPORT_JOINTS = (
    "left_wrist", "right_wrist", "left_ankle", "right_ankle",
    "left_foot", "right_foot", "left_knee", "right_knee",
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "pelvis", "spine1", "spine2", "spine3",
    "neck", "head", "left_collar", "right_collar",
)
TAUS_CM = (3.0, 5.0, 8.0)
OCC_FULL = 5           # current-method count that zeros visibility
OCC_SUBSAMPLE = 2500   # current-method vertex subsample (stage default)
OWNREF_FRONT_M = 0.005     # own-surface criterion: non-own must be this far in front of own skin
MULTIRAY_TAU_M = 0.05      # joint-centre tau used for the per-ray occlusion test
MULTIRAY_OFFSET_M = 0.015  # perpendicular offset of the 4 satellite rays
STABLE_DRIFT_M = 0.01      # frame-to-frame joint move below which pose is "stable" (flicker test)
_EPS = 1e-8


@torch.no_grad()
def _pose(mid, betas: torch.Tensor, q: torch.Tensor, scale: float, *, chunk: int = 64):
    """Pose all frames → ``(joints (N,J,3), vertices (N,V,3))`` in the world frame.

    Copies ``estimate_contacts._pose``: re-applies the per-person global ``scale``
    about the pelvis so the mesh sits at its stored metric size.
    """
    num_frames = q.shape[0]
    joints_out, verts_out = [], []
    for start in range(0, num_frames, chunk):
        end = min(start + chunk, num_frames)
        out = mid.forward(betas[None].expand(end - start, -1),
                          normalize_q(q[start:end], mid.num_joints))
        jw = out.joints_world.tensor()[..., :3]
        vw = out.vertices
        root = jw[:, :1, :]
        joints_out.append(root + scale * (jw - root))
        verts_out.append(root + scale * (vw - root))
    return torch.cat(joints_out, 0), torch.cat(verts_out, 0)


def _moller_trumbore(origin: torch.Tensor, dirs: torch.Tensor,
                     v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor):
    """Ray-triangle intersection for ``K`` rays vs ``F`` triangles (one shared origin).

    :param origin: ``(3,)`` common ray origin (camera centre).
    :param dirs: ``(K,3)`` unit ray directions.
    :param v0,v1,v2: ``(F,3)`` triangle vertices.
    :returns: ``(t_hit (K,F), hit (K,F) bool)`` — ``t`` is the along-ray distance.
    """
    num_rays, num_faces = dirs.shape[0], v0.shape[0]
    edge1 = v1 - v0                                            # (F,3)
    edge2 = v2 - v0                                            # (F,3)
    pvec = torch.linalg.cross(
        dirs[:, None, :].expand(-1, num_faces, -1),
        edge2[None, :, :].expand(num_rays, -1, -1))            # (K,F,3)
    det = (edge1[None] * pvec).sum(-1)                         # (K,F)
    inv_det = 1.0 / torch.where(det.abs() < _EPS, torch.full_like(det, _EPS), det)
    tvec = origin - v0                                         # (F,3)
    bary_u = (tvec[None] * pvec).sum(-1) * inv_det             # (K,F)
    qvec = torch.linalg.cross(tvec, edge1)                    # (F,3)
    bary_v = (dirs[:, None, :] * qvec[None]).sum(-1) * inv_det  # (K,F)
    t_hit = (edge2[None] * qvec[None]).sum(-1) * inv_det       # (K,F)
    hit = ((det.abs() >= _EPS) & (bary_u >= 0.0) & (bary_v >= 0.0)
           & (bary_u + bary_v <= 1.0) & (t_hit > _EPS))
    return t_hit, hit


def _perp_basis(dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two orthonormal directions perpendicular to each unit ray ``dirs (J,3)``."""
    up = torch.tensor([0.0, 0.0, 1.0], device=dirs.device).expand_as(dirs)
    alt = torch.tensor([0.0, 1.0, 0.0], device=dirs.device).expand_as(dirs)
    ref = torch.where((dirs * up).sum(-1, keepdim=True).abs() > 0.9, alt, up)
    e_u = torch.linalg.cross(dirs, ref)
    e_u = e_u / e_u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    e_v = torch.linalg.cross(dirs, e_u)
    e_v = e_v / e_v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return e_u, e_v


@torch.no_grad()
def exact_ray_cast(
    cam_centers: torch.Tensor,   # (N,3) world camera centres
    joints: torch.Tensor,        # (N,J,3) target world joint centres
    verts: torch.Tensor,         # (N,Vtot,3) world vertices of ALL people, concatenated
    faces: torch.Tensor,         # (Ftot,3) long faces of ALL people (offset per person)
    own_mask: torch.Tensor,      # (J,Ftot) bool — face is the target joint's own skin
) -> dict[str, np.ndarray]:
    """Single centre ray per ``(frame, joint)``; returns depth references ``(N,J)``.

    Returns ``joint_depth``, ``own_depth`` (nearest own-group hit, ``+inf`` = ray
    misses all own skin), and ``nonown_depth`` (nearest non-own hit, ``+inf`` = no
    occluder). One frame at a time to bound memory.
    """
    num_frames, num_joints = joints.shape[0], joints.shape[1]
    device = joints.device
    joint_depth = np.zeros((num_frames, num_joints), np.float32)
    own_depth = np.full((num_frames, num_joints), np.inf, np.float32)
    nonown_depth = np.full((num_frames, num_joints), np.inf, np.float32)
    for t in range(num_frames):
        origin = cam_centers[t]
        u = joints[t] - origin                                # (J,3)
        length = u.norm(dim=-1)                               # (J,)
        dirs = u / length.clamp_min(1e-9)[:, None]
        v0, v1, v2 = (verts[t][faces[:, k]] for k in range(3))
        t_hit, hit = _moller_trumbore(origin, dirs, v0, v1, v2)   # (J,F)
        own_hit = hit & own_mask
        nonown_hit = hit & ~own_mask
        own = torch.where(own_hit, t_hit, torch.full_like(t_hit, float("inf"))).amin(1)
        nonown = torch.where(nonown_hit, t_hit, torch.full_like(t_hit, float("inf"))).amin(1)
        joint_depth[t] = length.cpu().numpy()
        own_depth[t] = own.cpu().numpy()
        nonown_depth[t] = nonown.cpu().numpy()
    return dict(joint_depth=joint_depth, own_depth=own_depth, nonown_depth=nonown_depth)


@torch.no_grad()
def multi_ray_occlusion(
    cam_centers: torch.Tensor,   # (N,3)
    joints: torch.Tensor,        # (N,J,3)
    verts: torch.Tensor,         # (N,Vtot,3)
    faces: torch.Tensor,         # (Ftot,3)
    own_mask: torch.Tensor,      # (J,Ftot)
    *,
    tau_m: float = MULTIRAY_TAU_M,
    offset_m: float = MULTIRAY_OFFSET_M,
) -> dict[str, np.ndarray]:
    """5-ray (centre + 4 perpendicular offsets) joint-centre occlusion per ``(frame, joint)``.

    Each ray is occluded when its nearest non-own hit lies more than ``tau_m`` in
    front of the joint depth. Returns ``single_occ`` (centre ray, bool) and
    ``frac_occ`` (fraction of the 5 rays occluded), each ``(N,J)``.
    """
    num_frames, num_joints = joints.shape[0], joints.shape[1]
    device = joints.device
    single_occ = np.zeros((num_frames, num_joints), bool)
    frac_occ = np.zeros((num_frames, num_joints), np.float32)
    own5 = own_mask.repeat_interleave(5, dim=0)               # (5J,F)
    for t in range(num_frames):
        origin = cam_centers[t]
        u = joints[t] - origin                                # (J,3)
        length = u.norm(dim=-1)                               # (J,)
        dirs = u / length.clamp_min(1e-9)[:, None]            # (J,3)
        e_u, e_v = _perp_basis(dirs)
        # 5 target points per joint: centre + 4 perpendicular offsets, same depth.
        targets = torch.stack([
            joints[t],
            joints[t] + offset_m * e_u, joints[t] - offset_m * e_u,
            joints[t] + offset_m * e_v, joints[t] - offset_m * e_v], dim=1)  # (J,5,3)
        ray_u = targets - origin                              # (J,5,3)
        ray_dirs = (ray_u / ray_u.norm(dim=-1, keepdim=True).clamp_min(1e-9)).reshape(-1, 3)
        v0, v1, v2 = (verts[t][faces[:, k]] for k in range(3))
        t_hit, hit = _moller_trumbore(origin, ray_dirs, v0, v1, v2)   # (5J,F)
        nonown = torch.where(hit & ~own5, t_hit, torch.full_like(t_hit, float("inf"))).amin(1)
        nonown = nonown.reshape(num_joints, 5)                # (J,5)
        occ_ray = nonown < (length[:, None] - tau_m)          # (J,5) bool
        single_occ[t] = occ_ray[:, 0].cpu().numpy()
        frac_occ[t] = occ_ray.float().mean(1).cpu().numpy()
    return dict(single_occ=single_occ, frac_occ=frac_occ)


def _face_vertex_groups(faces: torch.Tensor, vertex_group: torch.Tensor) -> torch.Tensor:
    """Per-face vertex-group ids ``(F,3)`` for one person's target ownership."""
    return vertex_group[faces]


def _build_own_mask(faces_all: torch.Tensor, face_groups: torch.Tensor,
                    joint_group: torch.Tensor) -> torch.Tensor:
    """``(J,F)`` bool: face is the joint's own skin (ANY vertex in the joint's group)."""
    g0, g1, g2 = face_groups[:, 0], face_groups[:, 1], face_groups[:, 2]
    jg = joint_group[:, None]                                 # (J,1)
    return (jg == g0[None]) | (jg == g1[None]) | (jg == g2[None])


def _load_scene(pred_dir: Path, device: str):
    """Load kinematic_1 + transform for one scene; return the pieces the audit needs."""
    kin = dict(np.load(pred_dir / "human_optim" / "kinematic_1.npz", allow_pickle=True))
    tr = dict(np.load(pred_dir / "geometry" / "transform.npz", allow_pickle=True))
    num_frames, num_people = int(kin["num_frames"]), int(kin["num_people"])
    height, width = int(kin["video_height"]), int(kin["video_width"])
    valid_mask = np.asarray(kin["valid_mask"], dtype=bool)          # (P,N)
    extr = torch.as_tensor(tr["extrinsics"], dtype=torch.float32, device=device)
    intr = torch.as_tensor(tr["intrinsics_px_orig"], dtype=torch.float32, device=device)
    frame_indices = np.asarray(tr["frame_indices"], dtype=np.int64)
    return dict(kin=kin, num_frames=num_frames, num_people=num_people, height=height,
                width=width, valid_mask=valid_mask, extr=extr, intr=intr,
                frame_indices=frame_indices)


@torch.no_grad()
def audit_scene(pred_dir: Path, mid, device: str):
    """Run the current + exact occlusion methods over one scene; return every field."""
    scene = _load_scene(pred_dir, device)
    kin = scene["kin"]
    num_frames, num_people = scene["num_frames"], scene["num_people"]
    valid_mask = scene["valid_mask"]
    extr, intr = scene["extr"], scene["intr"]
    cam_centers = world_from_cam(extr)[:, :3, 3]                    # (N,3)

    joint_names = list(mid.joint_names)
    num_joints = mid.num_joints
    occ_group = torch.as_tensor(occlusion_group_ids(joint_names), device=device)  # (J,)
    faces = mid.faces.to(device).long()                            # (F,3)
    num_verts = mid.num_vertices

    people = []
    for p in range(num_people):
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        scale = float(kin["scale"][p])
        v_rest, j_rest = mid.forward_shape(betas[None])
        v2j = vertex_to_joint(v_rest[0], j_rest[0])                 # (V,)
        jw, vw = _pose(mid, betas, q, scale)                       # (N,J,3),(N,V,3)
        invalid = torch.as_tensor(~valid_mask[p], device=device)
        vw = torch.where(invalid[:, None, None], torch.full_like(vw, float("nan")), vw)
        people.append(dict(v2j=v2j, jw=jw, vw=vw))

    all_verts = torch.cat([person["vw"] for person in people], dim=1)   # (N, P*V, 3)
    all_faces = torch.cat([faces + p * num_verts for p in range(num_people)], dim=0)

    counts = np.zeros((num_people, num_frames, num_joints), np.float32)
    joint_depth = np.zeros((num_people, num_frames, num_joints), np.float32)
    own_depth = np.zeros((num_people, num_frames, num_joints), np.float32)
    nonown_depth = np.zeros((num_people, num_frames, num_joints), np.float32)
    single_occ = np.zeros((num_people, num_frames, num_joints), bool)
    frac_occ = np.zeros((num_people, num_frames, num_joints), np.float32)
    exact_seconds = multiray_seconds = 0.0

    for p, person in enumerate(people):
        v2j, jw, vw = person["v2j"], person["jw"], person["vw"]
        other_vw = None
        if num_people > 1:
            other_vw = torch.stack(
                [people[i]["vw"] for i in range(num_people) if i != p], dim=1)  # (N,Po,V,3)
        counts[p] = joint_occlusion_counts(
            vw, jw, cam_centers, v2j, occ_group,
            other_vertices=other_vw, perp_m=0.02, front_m=0.005,
            subsample=OCC_SUBSAMPLE).cpu().numpy()

        # Per-face vertex groups: target person uses its ownership; other people = -2.
        face_groups = torch.full((all_faces.shape[0], 3), -2, dtype=torch.long, device=device)
        target_slice = slice(p * faces.shape[0], (p + 1) * faces.shape[0])
        face_groups[target_slice] = _face_vertex_groups(faces, occ_group[v2j])
        own_mask = _build_own_mask(all_faces, face_groups, occ_group)   # (J,F)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cast = exact_ray_cast(cam_centers, jw, all_verts, all_faces, own_mask)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        exact_seconds += time.perf_counter() - t0
        joint_depth[p], own_depth[p], nonown_depth[p] = (
            cast["joint_depth"], cast["own_depth"], cast["nonown_depth"])

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mray = multi_ray_occlusion(cam_centers, jw, all_verts, all_faces, own_mask)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        multiray_seconds += time.perf_counter() - t0
        single_occ[p], frac_occ[p] = mray["single_occ"], mray["frac_occ"]

    gap = joint_depth - nonown_depth                               # (P,N,J), -inf when no non-own hit
    # frame-to-frame joint displacement (for the flicker test), per person.
    disp = np.full((num_people, num_frames, num_joints), np.inf, np.float32)
    for p in range(num_people):
        jw = people[p]["jw"].cpu().numpy()
        disp[p, 1:] = np.linalg.norm(jw[1:] - jw[:-1], axis=-1)

    proj = []
    for person in people:
        jw_cam = transform_points(person["jw"], extr)
        vw_cam = transform_points(person["vw"], extr)
        proj.append(dict(joints_px=project(jw_cam, intr).cpu().numpy(),
                         verts_px=project(vw_cam, intr).cpu().numpy()))

    return dict(count=counts, gap=gap, joint_depth=joint_depth, own_depth=own_depth,
                nonown_depth=nonown_depth, single_occ=single_occ, frac_occ=frac_occ,
                disp=disp, valid=valid_mask, num_frames=num_frames, num_people=num_people,
                height=scene["height"], width=scene["width"],
                frame_indices=scene["frame_indices"], joint_names=joint_names, proj=proj,
                exact_seconds=exact_seconds, multiray_seconds=multiray_seconds)


def _ownref_occluded(data: dict) -> np.ndarray:
    """Own-surface-reference occlusion ``(P,N,J)`` bool: non-own hit in front of own skin."""
    ref = np.where(np.isfinite(data["own_depth"]), data["own_depth"], data["joint_depth"])
    return data["nonown_depth"] < (ref - OWNREF_FRONT_M)


def _confusion(cur_occ: np.ndarray, exa_occ: np.ndarray, valid: np.ndarray,
               joint_names: list[str], label: str) -> dict:
    """Confusion of two boolean occlusion masks over valid joint-frames."""
    vmj = np.broadcast_to(valid[:, :, None], cur_occ.shape)
    cur = cur_occ & vmj
    exa = exa_occ & vmj
    n = int(vmj.sum())
    per_joint = {}
    name_to_idx = {nm: i for i, nm in enumerate(joint_names)}
    for name in REPORT_JOINTS:
        j = name_to_idx[name]
        cj = cur_occ[:, :, j] & valid
        ej = exa_occ[:, :, j] & valid
        nj = int(valid.sum())
        per_joint[name] = dict(
            n=nj,
            current_occ_rate=float(cj.sum() / max(nj, 1)),
            exact_occ_rate=float(ej.sum() / max(nj, 1)),
            both=int((cj & ej).sum()),
            only_current=int((cj & ~ej).sum()),
            only_exact=int((ej & ~cj).sum()))
    return dict(label=label, n=n, both=int((cur & exa).sum()),
                only_current=int((cur & ~exa & vmj).sum()),
                only_exact=int((exa & ~cur & vmj).sum()),
                neither=int((~cur & ~exa & vmj).sum()), per_joint=per_joint)


def _sensitivity(data: dict, valid: np.ndarray, joint_names: list[str]) -> dict:
    """Per-joint exact occlusion rate across tau + own-skin-miss + self-occlusion gaps."""
    gap = data["gap"]
    miss_own = ~np.isfinite(data["own_depth"])
    name_to_idx = {nm: i for i, nm in enumerate(joint_names)}
    out = {}
    for name in REPORT_JOINTS:
        j = name_to_idx[name]
        gj = gap[:, :, j][valid]
        pos = gj[gj > 0.0]
        finite = gj[np.isfinite(gj)]
        rates = {f"occ_rate_tau{int(t)}cm": float((gj > t / 100.0).mean()) for t in TAUS_CM}
        out[name] = dict(
            n=int(gj.size), any_hit_rate=float((gj > 0.0).mean()),
            miss_own_rate=float(miss_own[:, :, j][valid].mean()),
            median_gap_pos_cm=float(np.median(pos) * 100.0) if pos.size else float("nan"),
            p90_gap_cm=float(np.percentile(finite, 90) * 100.0) if finite.size else float("nan"),
            **rates)
    return out


def _multiray_stats(data: dict, valid: np.ndarray, joint_names: list[str]) -> dict:
    """Single- vs 5-ray disagreement + stable-pose flicker, pooled and per joint."""
    single = data["single_occ"]
    frac = data["frac_occ"]
    disp = data["disp"]
    name_to_idx = {nm: i for i, nm in enumerate(joint_names)}
    vmj = np.broadcast_to(valid[:, :, None], single.shape)
    disagree = (np.abs(single.astype(np.float32) - frac) > 0.5) & vmj

    def flicker(occ_bool: np.ndarray, mask_joint: int | None) -> float:
        toggles = eligible = 0
        P, N, J = occ_bool.shape
        js = range(J) if mask_joint is None else [mask_joint]
        for p in range(P):
            for t in range(1, N):
                if not (valid[p, t] and valid[p, t - 1]):
                    continue
                for j in js:
                    if disp[p, t, j] < STABLE_DRIFT_M:
                        eligible += 1
                        toggles += int(occ_bool[p, t, j] != occ_bool[p, t - 1, j])
        return float(toggles / eligible) if eligible else float("nan")

    five_occ = frac >= 0.5
    pooled = dict(
        disagree_rate=float(disagree.sum() / max(int(vmj.sum()), 1)),
        flicker_single=flicker(single, None),
        flicker_5ray=flicker(five_occ, None))
    per_joint = {}
    for name in REPORT_JOINTS:
        j = name_to_idx[name]
        vj = valid
        dj = (np.abs(single[:, :, j].astype(np.float32) - frac[:, :, j]) > 0.5) & vj
        per_joint[name] = dict(
            disagree_rate=float(dj.sum() / max(int(vj.sum()), 1)),
            flicker_single=flicker(single, j),
            flicker_5ray=flicker(five_occ, j))
    return dict(pooled=pooled, per_joint=per_joint)


def _examples(data: dict, scene: str, tau_cm: float = 5.0, k: int = 8):
    """Most confident current-vs-exact (joint-centre) disagreements, restricted to REPORT_JOINTS."""
    count, gap, valid = data["count"], data["gap"], data["valid"]
    joint_names = data["joint_names"]
    tau_m = tau_cm / 100.0
    report_mask = np.zeros(len(joint_names), dtype=bool)
    for name in REPORT_JOINTS:
        report_mask[joint_names.index(name)] = True
    vmj = np.broadcast_to(valid[:, :, None], count.shape) & report_mask[None, None, :]
    cur = (count >= OCC_FULL) & vmj
    exa = (gap > tau_m) & vmj

    def row(p, t, j) -> dict:
        g = float(gap[p, t, j])
        return dict(scene=scene, person=int(p), frame=int(t), joint=joint_names[j],
                    current_count=float(count[p, t, j]), exact_any_hit=bool(np.isfinite(g)),
                    exact_gap_cm=(float(g * 100.0) if np.isfinite(g) else None))

    out = {"only_current_H1": [], "only_exact_H2": []}
    idx = np.argwhere(cur & ~exa)
    if idx.size:
        score = -count[idx[:, 0], idx[:, 1], idx[:, 2]]
        for p, t, j in idx[np.argsort(score)[:k]]:
            out["only_current_H1"].append(row(p, t, j))
    idx = np.argwhere(exa & ~cur & (count < 1))
    if idx.size:
        score = -gap[idx[:, 0], idx[:, 1], idx[:, 2]]
        for p, t, j in idx[np.argsort(score)[:k]]:
            out["only_exact_H2"].append(row(p, t, j))
    return out


def _render_overlay(example: dict, scene_data: dict, pred_dir: Path, save_dir: Path, tag: str):
    """Render one disagreement overlay PNG (faint mesh scatter + marked joint)."""
    import cv2
    p, t, jname = example["person"], example["frame"], example["joint"]
    j = scene_data["joint_names"].index(jname)
    fi = int(scene_data["frame_indices"][t])
    video = pred_dir.parent.parent / "data" / f"{pred_dir.name}.mp4"
    if not video.exists():
        return None
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    verts_px = scene_data["proj"][p]["verts_px"][t]
    joints_px = scene_data["proj"][p]["joints_px"][t]
    finite = np.isfinite(verts_px).all(1)
    for x, y in verts_px[finite][::8].astype(int):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 1, (0, 200, 255), -1)
    jx, jy = joints_px[j]
    if np.isfinite(jx) and np.isfinite(jy):
        cv2.drawMarker(img, (int(jx), int(jy)), (255, 0, 0), cv2.MARKER_CROSS, 26, 3)
        cv2.circle(img, (int(jx), int(jy)), 12, (255, 0, 0), 2)
    gap = example["exact_gap_cm"]
    gap_str = f"{gap:+.1f}cm" if gap is not None else "no_hit(visible)"
    text = f"{jname} p{p} f{t}  cur_count={example['current_count']:.0f}  exact_gap={gap_str}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out_path = save_dir / f"{pred_dir.name}_{tag}_{jname}_p{p}_f{t}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", default="out", help="root holding out/<scene>/")
    ap.add_argument("--scenes", nargs="+",
                    default=["sport_climb", "boulder_1", "gtIGgs_y2nY_0024"])
    ap.add_argument("--save-dir", default="scratch/occlusion_audit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overlays", type=int, default=6)
    args = ap.parse_args(argv)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    mid = load_smplx_mid(None, device)

    report = {
        "config": dict(occ_full=OCC_FULL, occ_subsample=OCC_SUBSAMPLE,
                       taus_cm=list(TAUS_CM), ownref_front_m=OWNREF_FRONT_M,
                       multiray_tau_m=MULTIRAY_TAU_M, multiray_offset_m=MULTIRAY_OFFSET_M,
                       stable_drift_m=STABLE_DRIFT_M, own_rule="any-vertex-in-group"),
        "scenes": {}, "timing": {}}
    pooled = {"count": [], "gap": [], "ownref": [], "single": [], "frac": [],
              "disp": [], "valid": [], "own": [], "nonown": [], "jdepth": []}
    scene_blobs = {}
    for name in args.scenes:
        pred_dir = Path(args.predictions) / name
        print(f"[occlusion_audit] {name}: posing + casting …", flush=True)
        data = audit_scene(pred_dir, mid, device)
        scene_blobs[name] = (pred_dir, data)
        ownref = _ownref_occluded(data)
        np.savez(save_dir / f"{name}_occlusion.npz",
                 count=data["count"], gap=data["gap"], own_depth=data["own_depth"],
                 nonown_depth=data["nonown_depth"], joint_depth=data["joint_depth"],
                 single_occ=data["single_occ"], frac_occ=data["frac_occ"],
                 valid=data["valid"], joint_names=np.array(data["joint_names"]))
        report["timing"][name] = dict(
            exact_seconds=round(data["exact_seconds"], 3),
            multiray_seconds=round(data["multiray_seconds"], 3),
            num_frames=data["num_frames"], num_people=data["num_people"],
            faces=int(mid.faces.shape[0]),
            joint_frames=int(data["valid"].sum()) * mid.num_joints)
        cur_full = data["count"] >= OCC_FULL
        report["scenes"][name] = dict(
            num_people=data["num_people"], num_frames=data["num_frames"],
            confusion_jointcenter={
                f"tau{int(t)}cm": _confusion(cur_full, data["gap"] > t / 100.0,
                                             data["valid"], data["joint_names"], f"tau{int(t)}cm")
                for t in TAUS_CM},
            confusion_ownref=_confusion(cur_full, ownref, data["valid"],
                                        data["joint_names"], "ownref"),
            sensitivity=_sensitivity(data, data["valid"], data["joint_names"]),
            multiray=_multiray_stats(data, data["valid"], data["joint_names"]),
            examples=_examples(data, name))
        J = mid.num_joints
        pooled["count"].append(data["count"].reshape(-1, J))
        pooled["gap"].append(data["gap"].reshape(-1, J))
        pooled["ownref"].append(ownref.reshape(-1, J))
        pooled["single"].append(data["single_occ"].reshape(-1, J))
        pooled["frac"].append(data["frac_occ"].reshape(-1, J))
        pooled["disp"].append(data["disp"].reshape(-1, J))
        pooled["own"].append(data["own_depth"].reshape(-1, J))
        pooled["nonown"].append(data["nonown_depth"].reshape(-1, J))
        pooled["jdepth"].append(data["joint_depth"].reshape(-1, J))
        vmj = np.broadcast_to(data["valid"][:, :, None], data["count"].shape)
        pooled["valid"].append(vmj.reshape(-1, J))
        print(f"[occlusion_audit] {name}: exact {data['exact_seconds']:.2f}s + "
              f"multiray {data['multiray_seconds']:.2f}s "
              f"({data['num_frames']}f × {data['num_people']}p)", flush=True)

    joint_names = list(mid.joint_names)
    pool = {kk: np.concatenate(vv, 0)[None] for kk, vv in pooled.items()
            if kk not in ("valid",)}
    pool_valid = np.concatenate(pooled["valid"], 0)[:, 0][None]
    pool_data = dict(count=pool["count"], gap=pool["gap"], own_depth=pool["own"],
                     nonown_depth=pool["nonown"], joint_depth=pool["jdepth"],
                     single_occ=pool["single"].astype(bool), frac_occ=pool["frac"],
                     disp=pool["disp"], valid=pool_valid, joint_names=joint_names)
    cur_full = pool["count"] >= OCC_FULL
    report["pooled"] = dict(
        confusion_jointcenter={
            f"tau{int(t)}cm": _confusion(cur_full, pool["gap"] > t / 100.0,
                                         pool_valid, joint_names, f"tau{int(t)}cm")
            for t in TAUS_CM},
        confusion_ownref=_confusion(cur_full, pool["ownref"].astype(bool),
                                    pool_valid, joint_names, "ownref"),
        sensitivity=_sensitivity(pool_data, pool_valid, joint_names),
        multiray=_multiray_stats(pool_data, pool_valid, joint_names))

    overlay_paths = []
    ranked = sorted(scene_blobs.items(),
                    key=lambda kv: (kv[1][1]["count"] >= OCC_FULL).mean(), reverse=True)
    for name, (pred_dir, data) in ranked:
        ex = _examples(data, name)
        picks = [("H1", e) for e in ex["only_current_H1"][:4]]
        picks += [("H2", e) for e in ex["only_exact_H2"][:2]]
        for tag, e in picks:
            path = _render_overlay(e, data, pred_dir, save_dir, tag)
            if path is not None:
                overlay_paths.append(str(path))
            if len(overlay_paths) >= args.overlays:
                break
        if len(overlay_paths) >= args.overlays:
            break
    report["overlays"] = overlay_paths

    out_json = save_dir / "audit_report.json"
    with out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[occlusion_audit] wrote {out_json}")
    print(f"[occlusion_audit] overlays: {len(overlay_paths)} → {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
