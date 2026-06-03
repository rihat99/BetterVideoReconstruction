"""MHR → better_human SMPL conversion.

Ports the barycentric topology mapping + PyTorch Adam fitting from
`third_party/MHR/tools/mhr_smpl_conversion/` to `better_human.smpl.SMPL`.

Unlike the upstream tool (which targets `smplx.SMPL` / `smplx.SMPLX`), this
writes parameters for the robotics-style `better_human` SMPL: betas (10) plus
a 99-dim `q` vector (SE(3) root + 23 SO(3) joint quaternions). We also emit
the classic (axis-angle) parametrization for ecosystem compatibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# --- topology edges derived from the SMPL face array -------------------------


def _unique_edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    """Return unique undirected edges as (E, 2) long tensor of sorted vertex ids."""
    f = faces.reshape(-1, 3).long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    e, _ = e.sort(dim=1)
    e = torch.unique(e, dim=0)
    return e


# --- main converter ----------------------------------------------------------


class MhrToSmplConverter:
    """Fit better_human SMPL parameters to MHR bodies placed in camera frame.

    Camera frame reconstruction follows sam-3d-body convention: MHR vertices
    (body-local, cm) get y/z flipped, then `100 * pred_cam_t` is added to
    place them in the camera frame, then scaled to meters before fitting.
    """

    def __init__(
        self,
        mhr_model,
        smpl_model,
        mapping_npz: str | Path,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.mhr = mhr_model.to(self.device)
        self.smpl = smpl_model.to(self.device)
        self.batch_size = batch_size

        # barycentric mapping: each SMPL vertex = sum_i baryc[i] * mhr_face_vertex_i
        data = np.load(mapping_npz)
        self.triangle_ids = torch.as_tensor(
            data["triangle_ids"], dtype=torch.long, device=self.device
        )  # (6890,)
        self.baryc_coords = torch.as_tensor(
            data["baryc_coords"], dtype=torch.float32, device=self.device
        )  # (6890, 3)

        # MHR skinned-mesh faces (from pymomentum Character mesh; numpy on CPU)
        mhr_faces_np = np.asarray(self.mhr.character.mesh.faces, dtype=np.int64)
        self.mhr_faces = torch.as_tensor(mhr_faces_np, device=self.device)  # (F, 3)

        # SMPL edges (for edge-based fitting loss)
        self.smpl_edges = _unique_edges_from_faces(self.smpl.faces).to(self.device)

        self.num_face_expr = int(self.mhr.get_num_face_expression_blendshapes())

    # -- step 1: MHR forward + camera-frame reconstruction ------------------

    @torch.no_grad()
    def _mhr_to_camera_frame(
        self,
        mhr_model_params: torch.Tensor,  # (N, 204)
        shape_params: torch.Tensor,  # (N, 45)
        pred_cam_t: torch.Tensor,  # (N, 3), meters
    ) -> torch.Tensor:
        """Return MHR vertices in camera frame, in centimeters. (N, V, 3)."""
        N = mhr_model_params.shape[0]
        expr = torch.zeros(
            N, self.num_face_expr, device=self.device, dtype=torch.float32
        )
        out_chunks = []
        for s in range(0, N, self.batch_size):
            e = min(s + self.batch_size, N)
            verts, _ = self.mhr(
                shape_params[s:e].to(self.device),
                mhr_model_params[s:e].to(self.device),
                expr[s:e],
            )
            # sam-3d-body flips y/z before storing joints; mirror that here so
            # the barycentric result matches the camera convention of pred_cam_t.
            verts = verts.clone()
            verts[..., 1] *= -1
            verts[..., 2] *= -1
            verts_cam = verts + 100.0 * pred_cam_t[s:e].to(self.device)[:, None, :]
            out_chunks.append(verts_cam)
        return torch.cat(out_chunks, dim=0)

    # -- step 2: barycentric map to SMPL topology ----------------------------

    def _to_smpl_topology(self, mhr_verts_cm: torch.Tensor) -> torch.Tensor:
        """MHR camera-frame vertices (cm) → SMPL-topology verts (meters)."""
        # tri_verts: (N, 6890, 3, 3) where last 3 = 3 vertices of selected triangle
        tri = mhr_verts_cm[:, self.mhr_faces[self.triangle_ids], :]
        smpl_cm = (tri * self.baryc_coords[None, :, :, None]).sum(dim=2)
        return smpl_cm * 0.01  # meters

    # -- step 3: Adam fit ---------------------------------------------------

    def _smpl_forward(
        self,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> torch.Tensor:
        q = self.smpl.from_classic(betas, body_pose, transl, global_orient)
        return self.smpl(betas=betas, q=q).vertices

    def _fit_batch(
        self,
        target: torch.Tensor,  # (B, 6890, 3) meters, camera frame
        num_iters: int,
        lr_coarse: float,
        lr_fine: float,
    ) -> dict[str, torch.Tensor]:
        B = target.shape[0]
        dev = self.device

        betas = torch.zeros(B, 10, device=dev, requires_grad=True)
        body_pose = torch.zeros(B, 69, device=dev, requires_grad=True)
        global_orient = torch.zeros(B, 3, device=dev, requires_grad=True)
        transl = torch.zeros(B, 3, device=dev)

        edges = self.smpl_edges
        target_edges = target[:, edges[:, 1], :] - target[:, edges[:, 0], :]

        # coarse stage A: global_orient only, edge loss
        opt = torch.optim.Adam([global_orient], lr=lr_coarse)
        for _ in range(40):
            verts = self._smpl_forward(betas, body_pose, global_orient, transl)
            pred_edges = verts[:, edges[:, 1], :] - verts[:, edges[:, 0], :]
            loss = (pred_edges - target_edges).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # coarse stage B: add body_pose + betas, edge loss
        opt = torch.optim.Adam(
            [global_orient, body_pose, betas], lr=lr_coarse
        )
        for _ in range(40):
            verts = self._smpl_forward(betas, body_pose, global_orient, transl)
            pred_edges = verts[:, edges[:, 1], :] - verts[:, edges[:, 0], :]
            loss = (pred_edges - target_edges).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # bootstrap translation: align centroids
        with torch.no_grad():
            verts = self._smpl_forward(betas, body_pose, global_orient, transl)
            transl = transl + (target - verts).mean(dim=1)
        transl = transl.detach().requires_grad_(True)

        # fine stage: optimize all four params with edge → vertex loss schedule
        opt = torch.optim.Adam(
            [global_orient, body_pose, betas, transl], lr=lr_fine
        )
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[num_iters // 3, 2 * num_iters // 3], gamma=0.1
        )
        for epoch in range(num_iters):
            verts = self._smpl_forward(betas, body_pose, global_orient, transl)
            pred_edges = verts[:, edges[:, 1], :] - verts[:, edges[:, 0], :]
            edge_weight = 1.0 if epoch < 50 else 0.0
            edge_loss = (pred_edges - target_edges).abs().mean()
            vertex_loss = (verts - target).square().mean()
            loss = edge_weight * edge_loss + vertex_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        # final fit error (per-frame mean vertex distance in meters)
        with torch.no_grad():
            verts = self._smpl_forward(betas, body_pose, global_orient, transl)
            fit_err = ((verts - target) ** 2).sum(-1).sqrt().mean(dim=1)

        return {
            "betas": betas.detach(),
            "body_pose": body_pose.detach(),
            "global_orient": global_orient.detach(),
            "transl": transl.detach(),
            "fit_error": fit_err.detach(),
        }

    # -- public entry point -------------------------------------------------

    def convert(
        self,
        mhr_model_params: np.ndarray,  # (N, 204)
        shape_params: np.ndarray,  # (N, 45)
        pred_cam_t: np.ndarray,  # (N, 3)
        valid_mask: np.ndarray,  # (N,) bool
        num_iters: int = 300,
        lr_coarse: float = 0.1,
        lr_fine: float = 0.01,
    ) -> dict[str, np.ndarray]:
        N = int(mhr_model_params.shape[0])

        mp = torch.as_tensor(mhr_model_params, dtype=torch.float32)
        sp = torch.as_tensor(shape_params, dtype=torch.float32)
        ct = torch.as_tensor(pred_cam_t, dtype=torch.float32)

        valid_idx = np.flatnonzero(valid_mask)
        # Pre-allocate NaN outputs
        out = {
            "betas": np.full((N, 10), np.nan, dtype=np.float32),
            "body_pose": np.full((N, 69), np.nan, dtype=np.float32),
            "global_orient": np.full((N, 3), np.nan, dtype=np.float32),
            "transl": np.full((N, 3), np.nan, dtype=np.float32),
            "q": np.full((N, self.smpl.nq), np.nan, dtype=np.float32),
            "fit_error": np.full((N,), np.nan, dtype=np.float32),
        }
        if valid_idx.size == 0:
            return out

        mp_v = mp[valid_idx]
        sp_v = sp[valid_idx]
        ct_v = ct[valid_idx]

        # 1) reconstruct camera-frame MHR verts (cm)
        mhr_cam = self._mhr_to_camera_frame(mp_v, sp_v, ct_v)  # (Nv, V, 3) cm
        # 2) map to SMPL topology (m)
        target = self._to_smpl_topology(mhr_cam)  # (Nv, 6890, 3) m
        del mhr_cam

        # 3) fit in batches
        results: list[dict[str, torch.Tensor]] = []
        for s in tqdm(
            range(0, target.shape[0], self.batch_size),
            desc="Fitting SMPL",
        ):
            e = min(s + self.batch_size, target.shape[0])
            results.append(
                self._fit_batch(target[s:e], num_iters, lr_coarse, lr_fine)
            )

        # concat
        cat = {
            k: torch.cat([r[k] for r in results], dim=0) for k in results[0]
        }

        # classic → q (for convenience, so downstream doesn't have to recompute)
        with torch.no_grad():
            q_valid = self.smpl.from_classic(
                cat["betas"], cat["body_pose"], cat["transl"], cat["global_orient"]
            )

        # scatter back to N-length arrays
        out["betas"][valid_idx] = cat["betas"].cpu().numpy()
        out["body_pose"][valid_idx] = cat["body_pose"].cpu().numpy()
        out["global_orient"][valid_idx] = cat["global_orient"].cpu().numpy()
        out["transl"][valid_idx] = cat["transl"].cpu().numpy()
        out["q"][valid_idx] = q_valid.cpu().numpy()
        out["fit_error"][valid_idx] = cat["fit_error"].cpu().numpy()
        return out
