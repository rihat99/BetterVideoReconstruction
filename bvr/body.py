"""The one place BetterVideoReconstruction builds a human body (swap-ready seam).

Wraps BetterHuman's SMPL-X family (``better_human.Body`` → ``Identity`` →
``HumanState``; 52 native joints, ``q (N, 211)``) behind the small interface the
kinematic / contact / scale stages call, so a future body swap (MHR) only touches
this file. The wrapper reproduces the exact surface the stages used from the
previous ``SMPLXMid`` body:

* :meth:`Body.forward` — pose ``q`` → world joints + vertices + the contact
  frames (differentiable);
* :meth:`Body.forward_shape` — the neutral-pose shaped mesh + rest joints;
* :meth:`Body.from_classic` / :meth:`Body.to_classic` — classic SMPL-X ↔ ``q``;
* the static topology attributes (``faces``, dense ``lbs_weights``,
  ``joint_names``, landmark tables, ``parent_tree``, ``num_joints``, ``nq`` …).

Joint order is native ``better_human`` order (pelvis, 21 body, 15+15 fingers),
identical to the stored ``joint_names`` — so ``q`` is drop-in and no remap is
needed. Shape is baked lazily (once per distinct ``betas``) and reused across a
person's optimization.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import replace as _dataclass_replace
from pathlib import Path

import torch
from better_robot.lie import so3

import better_human as bh
from better_human.families import tables
from better_human.families.assets import load_density_table
from better_human.families.classic import SMPLXClassic, smplx_from_classic, smplx_to_classic

#: The BetterHuman checkout the installed (editable) package comes from
#: (``src/better_human/__init__.py`` → repo root): its ``models/`` holds the licensed
#: weights and ``studio_data/`` the authored contact frames.
_BETTER_HUMAN_ROOT = Path(bh.__file__).resolve().parents[2]


def models_dir() -> Path:
    """BetterHuman's licensed weights: ``$BETTERHUMAN_MODELS_DIR``, else the checkout's ``models/``."""
    env = os.environ.get("BETTERHUMAN_MODELS_DIR")
    return Path(env).expanduser() if env else _BETTER_HUMAN_ROOT / "models"


#: The authored **contact frames** — named mesh points baked onto the body as
#: BetterRobot frames. ``better_contacts.json`` is the full-body set (35 frames:
#: per-foot heel / ball_inner / ball_outer / big_toe / toe_above, hands, knees,
#: elbows, forearms, shoulders, sit bones, hips, chest, sacrum, back, head) that
#: extends the original 12-frame climbing set (``contacts.json``). Each frame's
#: origin is a template vertex and its local z axis is that vertex's outward
#: rest-pose surface normal, so forward kinematics reports both a contact point
#: and the direction the skin faces there. Contact detection lives entirely on
#: these frames (see ``docs/contacts.md``).
_DEFAULT_CONTACT_FRAMES = (
    _BETTER_HUMAN_ROOT / "studio_data" / "frames" / "smplx_neutral" / "better_contacts.json"
)


def frame_normals(frames7: torch.Tensor) -> torch.Tensor:
    """Outward world normal ``(..., F, 3)`` of contact-frame poses ``(..., F, 7)``.

    A contact frame's local z axis is baked as the outward mesh normal at its
    vertex, so the posed world normal is the third column of the frame's rotation
    matrix — no mesh work per frame.

    :param frames7: contact-frame world poses, ``[translation, quaternion xyzw]``.
    """
    return so3.to_matrix(frames7[..., 3:7])[..., :, 2]


class _JointsWorld:
    """World joint placements ``(..., 52, 7)`` — position + scalar-last quaternion.

    BetterRobot's universe joint (row 0) is dropped; the 52 rows are in native
    ``better_human`` joint order, matching :attr:`Body.joint_names`.
    """

    def __init__(self, placements: torch.Tensor) -> None:
        self._placements = placements

    def tensor(self) -> torch.Tensor:
        """Return the ``(..., 52, 7)`` position + quaternion placements."""
        return self._placements

    def matrix(self) -> torch.Tensor:
        """Return the ``(..., 52, 4, 4)`` homogeneous joint transforms."""
        rotation = so3.to_matrix(self._placements[..., 3:7])
        shape = (*self._placements.shape[:-1], 4, 4)
        matrix = self._placements.new_zeros(shape)
        matrix[..., :3, :3] = rotation
        matrix[..., :3, 3] = self._placements[..., :3]
        matrix[..., 3, 3] = 1.0
        return matrix


class _Posed:
    """One posed body: world joints + vertices + contact frames."""

    def __init__(self, vertices: torch.Tensor, joint_placements: torch.Tensor,
                 frame_placements: torch.Tensor, *, model=None, kinematics=None) -> None:
        #: Posed world vertices with shape ``(..., V, 3)``.
        self.vertices = vertices
        #: The BetterRobot model this pose was computed on and its FK
        #: ``KinematicState``, so a caller can take Jacobians of the same pose
        #: (``frame_jacobian(model, kinematics, frame=…)``, e.g. contact-frame
        #: velocities) without running FK again.
        self.model = model
        self.kinematics = kinematics
        #: World joint placements as a :class:`_JointsWorld` (``(..., 52, 7)``).
        self.joints_world = _JointsWorld(joint_placements)
        #: World joint positions with shape ``(..., 52, 3)``.
        self.joints = joint_placements[..., :3]
        #: Contact-frame world poses ``(..., F, 7)`` (position + scalar-last
        #: quaternion), in :attr:`Body.contact_frame_names` order. Unscaled, like
        #: the joints — a stage that re-applies a per-person scale about the
        #: pelvis scales the frame POSITIONS only (a uniform scale leaves the
        #: rotations, and therefore :func:`frame_normals`, untouched).
        self.frames_world = frame_placements


class Body:
    """Thin SMPL-X body wrapper over BetterHuman's :class:`better_human.Body`.

    :param base: a :class:`better_human.Body` of the SMPL-X family (``face=False``,
        ``hands=True``, ``num_betas=10``) built with contact frames.
    :param frames_allow: contact-frame names to keep (authored order preserved);
        ``None`` keeps every frame the body carries. The filter is the one place
        a recipe restricts where contact may live (e.g. the climb-wall rule:
        hands + feet only) — everything downstream (detection, labels, forces)
        simply sees a smaller frame set.
    :param sole_offsets: metres to push a contact frame out along its outward
        normal, keyed by frame stem (``"heel"`` covers ``heel_left`` and
        ``heel_right``). A shod foot touches the ground through its sole, and a
        shoe's heel stack is thicker than its forefoot, so the bare-skin heel frame
        rides a few centimetres above the surface the foot is actually standing
        on; the offset puts the frame on the sole. Applied in :meth:`forward`, so
        every consumer — detection, penetration, attraction, the force solve's
        lever arms — sees the same shod point. Unscaled metres, like the frames.
    """

    def __init__(self, base: bh.Body, frames_allow: Sequence[str] | None = None,
                 sole_offsets: Mapping[str, float] | None = None) -> None:
        if base.contact_frames is None:
            raise ValueError("body was built without contact frames; pass contact_frames= to Body")
        self._base = base
        self._dtype = base.dtype
        self._device = base.device

        #: Triangle indices with shape ``(F, 3)`` (long).
        self.faces = base.faces
        #: Native LBS joint names in weight-file order (length 52).
        self.joint_names = tuple(base.joint_names)
        #: Number of native joints (52).
        self.num_joints = len(self.joint_names)
        #: Configuration width (211): free-flyer root + 51 spherical joints.
        self.nq = base.model.nq
        #: Number of template vertices (10475).
        self.num_vertices = base.num_vertices
        #: Unshaped native template vertices with shape ``(V, 3)``.
        self.vertices_template = base.spec.vertices
        #: OpenPose-style landmark keypoint names, aligned with ``frame_vertex_ids``.
        self.frame_names = tuple(tables.SMPLX.keypoint_names)
        #: Template vertex index of each landmark keypoint, aligned with ``frame_names``.
        self.frame_vertex_ids = tuple(tables.SMPLX.keypoint_vertex_ids)
        #: Dense LBS skinning matrix with shape ``(V, 52)``.
        self.lbs_weights = base.skin_matrix
        #: Tissue density per body part ``(52,)`` in kg/m³, native joint order — what
        #: every baked inertia (and so every force in newtons) follows from.
        self.density = base.density
        #: Parent index per joint as ``(1, 52)`` (``[0, i]`` = parent of joint ``i``).
        self.parent_tree = torch.tensor(
            [base.parents], dtype=torch.long, device=self._device
        )

        spec = base.contact_frames
        if frames_allow is not None:
            allowed = set(frames_allow)
            unknown = allowed.difference(spec.names)
            if unknown:
                raise ValueError(f"unknown contact frames in allowlist: {sorted(unknown)}")
            keep = [i for i, name in enumerate(spec.names) if name in allowed]
            spec = _dataclass_replace(
                spec,
                names=tuple(spec.names[i] for i in keep),
                vertex_ids=tuple(spec.vertex_ids[i] for i in keep),
                parent_joints=tuple(spec.parent_joints[i] for i in keep),
                frame_ids=tuple(spec.frame_ids[i] for i in keep),
            )
        #: The resolved :class:`better_human.core.contact_frames.ContactFrames` spec.
        self.contact_frames = spec
        #: Contact-frame names in authored order, e.g. ``"big_toe_left"`` (length F).
        self.contact_frame_names = tuple(self.contact_frames.names)
        #: Parent joint of each contact frame, as a native 52-joint index (length F).
        self.contact_frame_parents = tuple(self.contact_frames.parent_joints)
        #: Template mesh vertex carrying each contact frame (length F).
        self.contact_frame_vertices = tuple(self.contact_frames.vertex_ids)
        # Rows of BetterRobot's frame table these frames occupy (53… after the
        # auto-generated per-body frames).
        self._contact_frame_ids = torch.tensor(
            self.contact_frames.frame_ids, dtype=torch.long, device=self._device
        )
        #: Those rows as plain ints (length F), for BetterRobot's per-frame Jacobians.
        self.contact_frame_ids = tuple(int(i) for i in self.contact_frames.frame_ids)
        #: Outward offset of each contact frame in metres ``(F,)``, from ``sole_offsets``.
        self.contact_frame_offsets = torch.zeros(
            len(self.contact_frame_names), dtype=self._dtype, device=self._device)
        if sole_offsets:
            stems = {n.rsplit("_", 1)[0] if n.endswith(("_left", "_right")) else n
                     for n in self.contact_frame_names}
            unknown = set(sole_offsets).difference(stems)
            if unknown:
                raise ValueError(f"sole offsets for frames this body does not carry: {sorted(unknown)}")
            for i, name in enumerate(self.contact_frame_names):
                stem = name.rsplit("_", 1)[0] if name.endswith(("_left", "_right")) else name
                self.contact_frame_offsets[i] = float(sole_offsets.get(stem, 0.0))

        self._shaped_cache: tuple[torch.Tensor, bh.Identity] | None = None

    def shaped(self, betas: torch.Tensor, *, fresh: bool = False) -> bh.Identity:
        """Return the body shaped at ``betas (..., 10)``, baking once and caching.

        The returned :class:`better_human.Identity` carries the BetterRobot model
        (``.model``, with the baked inertias in ``.model.values.body_inertias``) and
        FK (``.forward``) the dynamics solves need; a future MHR body implements the
        same contract. Shape baking (``blend_shapes`` + joint regression + inertia)
        is the expensive part, so a single-slot cache reuses the shaped body while
        ``betas`` is unchanged — the common case across one person's optimization.

        ``fresh`` bakes without touching the cache. Needed when ``betas`` is being
        optimized: the bake is differentiable, but a cached body carries the
        autograd graph of the iteration that baked it, and reusing it after that
        graph was freed raises.
        """
        betas = betas.to(dtype=self._dtype, device=self._device)
        if fresh:
            return self._base.bake(betas=betas)
        cached = self._shaped_cache
        if cached is not None and cached[0].shape == betas.shape and torch.equal(cached[0], betas):
            return cached[1]
        shaped = self._base.bake(betas=betas)
        self._shaped_cache = (betas.clone(), shaped)
        return shaped

    def forward(self, betas: torch.Tensor, q: torch.Tensor, *, fresh: bool = False) -> _Posed:
        """Pose ``q`` on the body shaped at ``betas``.

        :param betas: shape coefficients ``(..., 10)`` (constant across the batch
            in the optimizer; per-frame is allowed).
        :param q: configuration ``(..., 211)`` with unit quaternions.
        :param fresh: bake the shape outside the cache (see :meth:`shaped`); pass
            it whenever ``betas`` requires grad.
        :returns: a :class:`_Posed` with ``.vertices (..., V, 3)``,
            ``.joints_world`` (``(..., 52, 7)``), ``.joints (..., 52, 3)`` and
            ``.frames_world (..., F, 7)``.
        """
        shaped = self.shaped(betas, fresh=fresh)
        state = shaped.forward(q.to(dtype=self._dtype, device=self._device))
        # Contact frames ride the joint poses already computed by the FK, so placing
        # them is a handful of rigid composes — differentiable through ``q``.
        frames = state.frame_poses[..., self._contact_frame_ids, :]
        if bool((self.contact_frame_offsets != 0).any()):
            shifted = frames[..., :3] + self.contact_frame_offsets[:, None] * frame_normals(frames)
            frames = torch.cat([shifted, frames[..., 3:]], dim=-1)
        return _Posed(state.vertices, state.joint_poses, frames,
                      model=state.model, kinematics=state.kinematics)

    def forward_shape(self, betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the neutral-pose ``(v_shaped (..., V, 3), rest_joints (..., 52, 3))``.

        Absolute (un-centered) frame — the pelvis sits at its regressed rest
        position — matching the old ``forward_shape``. Consumers use these only for
        nearest-joint assignment (translation-invariant) or as a rest mesh for
        browser LBS (invariant to a shared translation of vertices and joints).
        """
        shaped = self.shaped(betas)
        pelvis = shaped.pelvis[..., None, :]
        v_shaped = shaped.v_shaped + pelvis
        rest_joints = shaped.rest_joints + pelvis
        return v_shaped, rest_joints

    def from_classic(
        self,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        transl: torch.Tensor,
        global_orient: torch.Tensor,
        left_hand_pose: torch.Tensor,
        right_hand_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Convert classic SMPL-X params → ``q (..., 211)`` (full axis-angle hands).

        Hands are full 45-D axis-angle deltas (``num_pca_comps=None``), matching the
        converter's stored parameters.
        """
        shaped = self.shaped(betas)
        prepare = lambda tensor: tensor.to(dtype=self._dtype, device=self._device)
        classic = SMPLXClassic(
            global_orient=prepare(global_orient),
            body_pose=prepare(body_pose),
            transl=prepare(transl),
            left_hand_pose=prepare(left_hand_pose),
            right_hand_pose=prepare(right_hand_pose),
            num_pca_comps=None,
        )
        q, _stream = smplx_from_classic(shaped, classic)
        return q

    def to_classic(self, betas: torch.Tensor, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert ``q (..., 211)`` → classic SMPL-X params keyed for the stages.

        Returns a dict with ``betas`` (echoed), ``global_orient``, ``translation``,
        ``body_pose``, ``left_hand_pose``, ``right_hand_pose`` (full axis-angle hands).
        """
        shaped = self.shaped(betas)
        classic = smplx_to_classic(shaped, q.to(dtype=self._dtype, device=self._device),
                                   num_pca_comps=None)
        return {
            "betas": betas,
            "global_orient": classic.global_orient,
            "translation": classic.transl,
            "body_pose": classic.body_pose,
            "left_hand_pose": classic.left_hand_pose,
            "right_hand_pose": classic.right_hand_pose,
        }


def load_body(model_path: str | Path | None = None, device: str | torch.device = "cpu",
              frames_allow: Sequence[str] | None = None,
              density: str | float = "Dempster",
              sole_offsets: Mapping[str, float] | None = None) -> Body:
    """Load the SMPL-X body used throughout the pipeline, contact frames baked on.

    :param model_path: path to a ``SMPLX_*.npz`` weight file; defaults to the
        neutral model under :func:`models_dir`.
    :param device: torch device for the model tensors.
    :param frames_allow: contact-frame names to keep; ``None`` keeps all.
    :param sole_offsets: outward offset per contact-frame stem, metres (see :class:`Body`).
    :param density: tissue density each body part's inertia is baked with — a
        table name from BetterHuman's ``body_densities.json`` (``"Dempster"``,
        ``"Clauser"``, ``"Chandler"``: one value per SMPL segment, fingers take
        the hand's) or a single uniform kg/m³ value. Mass, and so every force in
        newtons, follows from it.
    """
    path = Path(model_path) if model_path is not None else models_dir() / "smplx" / "SMPLX_NEUTRAL.npz"
    spec = bh.families.smplx(path, num_betas=10, hands=True, face=False, device=device)
    if isinstance(density, str):
        density = load_density_table(
            "body_densities.json", density,
            [spec.joint_names[j] for j in spec.parts.joints], spec.vertices)
    base = bh.Body(spec, density=density, contact_frames=_DEFAULT_CONTACT_FRAMES)
    return Body(base, frames_allow=frames_allow, sole_offsets=sole_offsets)
