"""MHR → SMPL-X conversion using the standard ``smplx`` PyPI package.

Mirrors :mod:`tools.mhr_to_smpl.converter` but targets SMPL-X (10475 vertices,
21 body joints, 15 + 15 hand joints, separate jaw/eye/expression channels).
The ``better_human.smpl.SMPLX`` implementation is incomplete (no jaw/eye/face
deformation) so we lean on pip ``smplx`` here as a temporary fix; downstream
``q`` packing for BetterRobot is intentionally deferred until we have a
matching kinematic chain in BetterRobot.

The fitting pipeline matches the SMPL converter's three-stage Adam:

  1. 40 iter — global_orient only, edge loss
  2. 40 iter — + body_pose + betas, edge loss
  3. ``num_iters`` iter — + transl + left/right hand pose, edge → vertex loss

Jaw/eye/expression are left at zero (MHR doesn't predict them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def _unique_edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    """Return unique undirected edges as (E, 2) long tensor of sorted vertex ids."""
    f = faces.reshape(-1, 3).long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    e, _ = e.sort(dim=1)
    e = torch.unique(e, dim=0)
    return e


class MhrToSmplxConverter:
    """Fit pip ``smplx.SMPLX`` parameters to MHR bodies in camera frame."""

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15

    def __init__(
        self,
        mhr_model: Any,
        smplx_model_factory: Any,
        mapping_npz: str | Path,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> None:
        """``smplx_model_factory(batch_size: int)`` returns a fresh SMPL-X model.

        We have to rebuild the body model when the batch size changes — pip
        ``smplx`` pre-allocates several buffers (e.g. ``lmk_faces_idx``) of
        shape ``(batch_size, ...)`` and does not broadcast at forward time.
        """
        self.device = torch.device(device)
        self.mhr = mhr_model.to(self.device)
        self.batch_size = batch_size
        self._smplx_factory = smplx_model_factory
        # Build a single template instance to read static buffers (faces).
        self._template = smplx_model_factory(batch_size).to(self.device)
        self._cache: dict[int, Any] = {batch_size: self._template}

        data = np.load(mapping_npz)
        self.triangle_ids = torch.as_tensor(
            data["triangle_ids"], dtype=torch.long, device=self.device
        )
        self.baryc_coords = torch.as_tensor(
            data["baryc_coords"], dtype=torch.float32, device=self.device
        )

        mhr_faces_np = np.asarray(self.mhr.character.mesh.faces, dtype=np.int64)
        self.mhr_faces = torch.as_tensor(mhr_faces_np, device=self.device)

        smplx_faces_np = np.asarray(self._template.faces, dtype=np.int64)
        self.smplx_edges = _unique_edges_from_faces(
            torch.as_tensor(smplx_faces_np)
        ).to(self.device)

        self.num_face_expr = int(self.mhr.get_num_face_expression_blendshapes())

    def _get_smplx(self, B: int):
        if B not in self._cache:
            self._cache[B] = self._smplx_factory(B).to(self.device)
        return self._cache[B]

    @property
    def faces(self) -> np.ndarray:
        return np.asarray(self._template.faces, dtype=np.int64)

    # -- step 1: MHR forward + camera-frame reconstruction ------------------

    @torch.no_grad()
    def _mhr_to_camera_frame(
        self,
        mhr_model_params: torch.Tensor,
        shape_params: torch.Tensor,
        pred_cam_t: torch.Tensor,
    ) -> torch.Tensor:
        """Return MHR vertices in camera frame, in centimeters. (N, V, 3)."""
        N = mhr_model_params.shape[0]
        expr = torch.zeros(
            N, self.num_face_expr, device=self.device, dtype=torch.float32,
        )
        out_chunks = []
        for s in range(0, N, self.batch_size):
            e = min(s + self.batch_size, N)
            verts, _ = self.mhr(
                shape_params[s:e].to(self.device),
                mhr_model_params[s:e].to(self.device),
                expr[s:e],
            )
            verts = verts.clone()
            verts[..., 1] *= -1
            verts[..., 2] *= -1
            verts_cam = verts + 100.0 * pred_cam_t[s:e].to(self.device)[:, None, :]
            out_chunks.append(verts_cam)
        return torch.cat(out_chunks, dim=0)

    # -- step 2: barycentric map to SMPL-X topology --------------------------

    def _to_smplx_topology(self, mhr_verts_cm: torch.Tensor) -> torch.Tensor:
        """MHR camera-frame vertices (cm) → SMPL-X-topology verts (meters)."""
        tri = mhr_verts_cm[:, self.mhr_faces[self.triangle_ids], :]
        smplx_cm = (tri * self.baryc_coords[None, :, :, None]).sum(dim=2)
        return smplx_cm * 0.01

    # -- step 3: Adam fit ---------------------------------------------------

    def _smplx_forward(
        self,
        smplx,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
        left_hand_pose: torch.Tensor,
        right_hand_pose: torch.Tensor,
    ) -> torch.Tensor:
        B = betas.shape[0]
        zeros3 = torch.zeros(B, 3, device=self.device)
        zeros10 = torch.zeros(B, 10, device=self.device)
        return smplx(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=zeros10,
            return_verts=True,
        ).vertices

    def _fit_batch(
        self,
        target: torch.Tensor,
        num_iters: int,
        lr_coarse: float,
        lr_fine: float,
    ) -> dict[str, torch.Tensor]:
        """Fit SMPL-X to ``target`` (B, V, 3) in metric meters.

        Mirrors the Meta reference (``MHR/tools/mhr_smpl_conversion/
        pytorch_fitting.py:_optimize_smpl``) closely — without that
        pre-centering plus the four-stage init we got stuck in flipped
        shoulder/wrist basins on poses with hands behind the head
        (e.g. ``core_workout_man``: left wrist axis-angle |aa|=235°
        before this fix). Centering the target removes the long
        translation walk so edge loss can shape the body before
        translation/vertex loss takes over.
        """
        B = target.shape[0]
        dev = self.device
        smplx = self._get_smplx(B)

        # Center target around origin so the T-pose starting body overlaps
        # the target by default — the SMPL-X template lives at the origin.
        target_center = 0.5 * (
            target.amin(dim=1, keepdim=True) + target.amax(dim=1, keepdim=True)
        )                                                  # (B, 1, 3)
        target_centered = target - target_center

        betas = torch.zeros(B, 10, device=dev, requires_grad=True)
        body_pose = torch.zeros(B, self.NUM_BODY_JOINTS * 3, device=dev, requires_grad=True)
        global_orient = torch.zeros(B, 3, device=dev, requires_grad=True)
        transl = torch.zeros(B, 3, device=dev)
        lh_pose = torch.zeros(B, self.NUM_HAND_JOINTS * 3, device=dev, requires_grad=True)
        rh_pose = torch.zeros(B, self.NUM_HAND_JOINTS * 3, device=dev, requires_grad=True)

        edges = self.smplx_edges
        target_edges = (
            target_centered[:, edges[:, 1], :]
            - target_centered[:, edges[:, 0], :]
        )

        def fwd():
            return self._smplx_forward(
                smplx, betas, body_pose, global_orient, transl, lh_pose, rh_pose,
            )

        def edge_loss(verts):
            pred_edges = verts[:, edges[:, 1], :] - verts[:, edges[:, 0], :]
            return (pred_edges - target_edges).abs().mean()

        # Stage A — only global_orient: align gross orientation against the
        # T-pose body before any pose parameters move.
        opt = torch.optim.Adam([global_orient], lr=lr_coarse)
        for _ in range(40):
            opt.zero_grad()
            edge_loss(fwd()).backward()
            opt.step()

        # Stage B — + body_pose + betas: gives the body 80 iters (Meta's
        # default) so shoulders/elbows have time to settle before hand
        # joints start fighting them. With only 40 iters here, our previous
        # converter stranded the chain in symmetric local minima
        # (left-wrist 235° on core_workout_man).
        opt = torch.optim.Adam([global_orient, body_pose, betas], lr=lr_coarse)
        for _ in range(80):
            opt.zero_grad()
            edge_loss(fwd()).backward()
            opt.step()

        # Stage C — + hands: hand-pose channels are decoupled from body
        # rotations once shoulders are roughly right.
        opt = torch.optim.Adam(
            [global_orient, body_pose, betas, lh_pose, rh_pose], lr=lr_coarse,
        )
        for _ in range(40):
            opt.zero_grad()
            edge_loss(fwd()).backward()
            opt.step()

        # Initialize translation so the centered prediction overlaps the
        # centered target, then fine-tune everything against vertex loss.
        with torch.no_grad():
            transl = transl + (target_centered - fwd()).mean(dim=1)
        transl = transl.detach().requires_grad_(True)

        opt = torch.optim.Adam(
            [global_orient, body_pose, betas, transl, lh_pose, rh_pose], lr=lr_fine,
        )
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[num_iters // 3, 2 * num_iters // 3], gamma=0.1,
        )
        for epoch in range(num_iters):
            verts = fwd()
            edge_weight = 1.0 if epoch < 50 else 0.0
            e = edge_loss(verts) if edge_weight > 0 else verts.new_zeros(())
            v = (verts - target_centered).square().mean()
            loss = edge_weight * e + v
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        # Move the fitted body back into the original frame: translation
        # was learned in the centered frame, so we add the center back.
        with torch.no_grad():
            transl_world = transl + target_center.squeeze(1)
            verts = self._smplx_forward(
                smplx, betas, body_pose, global_orient, transl_world,
                lh_pose, rh_pose,
            )
            fit_err = ((verts - target) ** 2).sum(-1).sqrt().mean(dim=1)

        return {
            "betas": betas.detach(),
            "body_pose": body_pose.detach(),
            "global_orient": global_orient.detach(),
            "transl": transl_world.detach(),
            "left_hand_pose": lh_pose.detach(),
            "right_hand_pose": rh_pose.detach(),
            "fit_error": fit_err.detach(),
        }

    # -- public entry point -------------------------------------------------

    def convert(
        self,
        mhr_model_params: np.ndarray,
        shape_params: np.ndarray,
        pred_cam_t: np.ndarray,
        valid_mask: np.ndarray,
        num_iters: int = 300,
        lr_coarse: float = 0.1,
        lr_fine: float = 0.01,
    ) -> dict[str, np.ndarray]:
        N = int(mhr_model_params.shape[0])
        nb = self.NUM_BODY_JOINTS * 3
        nh = self.NUM_HAND_JOINTS * 3

        mp = torch.as_tensor(mhr_model_params, dtype=torch.float32)
        sp = torch.as_tensor(shape_params, dtype=torch.float32)
        ct = torch.as_tensor(pred_cam_t, dtype=torch.float32)

        valid_idx = np.flatnonzero(valid_mask)
        out = {
            "betas": np.full((N, 10), np.nan, dtype=np.float32),
            "body_pose": np.full((N, nb), np.nan, dtype=np.float32),
            "global_orient": np.full((N, 3), np.nan, dtype=np.float32),
            "transl": np.full((N, 3), np.nan, dtype=np.float32),
            "left_hand_pose": np.full((N, nh), np.nan, dtype=np.float32),
            "right_hand_pose": np.full((N, nh), np.nan, dtype=np.float32),
            "jaw_pose": np.zeros((N, 3), dtype=np.float32),
            "leye_pose": np.zeros((N, 3), dtype=np.float32),
            "reye_pose": np.zeros((N, 3), dtype=np.float32),
            "expression": np.zeros((N, 10), dtype=np.float32),
            "fit_error": np.full((N,), np.nan, dtype=np.float32),
        }
        if valid_idx.size == 0:
            return out

        mp_v = mp[valid_idx]
        sp_v = sp[valid_idx]
        ct_v = ct[valid_idx]

        mhr_cam = self._mhr_to_camera_frame(mp_v, sp_v, ct_v)
        target = self._to_smplx_topology(mhr_cam)
        del mhr_cam

        results: list[dict[str, torch.Tensor]] = []
        for s in tqdm(range(0, target.shape[0], self.batch_size), desc="Fitting SMPL-X"):
            e = min(s + self.batch_size, target.shape[0])
            results.append(self._fit_batch(target[s:e], num_iters, lr_coarse, lr_fine))

        cat = {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}

        out["betas"][valid_idx] = cat["betas"].cpu().numpy()
        out["body_pose"][valid_idx] = cat["body_pose"].cpu().numpy()
        out["global_orient"][valid_idx] = cat["global_orient"].cpu().numpy()
        out["transl"][valid_idx] = cat["transl"].cpu().numpy()
        out["left_hand_pose"][valid_idx] = cat["left_hand_pose"].cpu().numpy()
        out["right_hand_pose"][valid_idx] = cat["right_hand_pose"].cpu().numpy()
        out["fit_error"][valid_idx] = cat["fit_error"].cpu().numpy()
        return out
