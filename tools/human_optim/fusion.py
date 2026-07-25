"""Fuse the geometric contact labels with an image model's per-limb predictions.

Two witnesses can say whether a limb touches the scene:

* the **geometric** one this pipeline already has — wall distance, stillness and a
  static support-force solve. It writes a label plus a *label confidence* in
  ``[0, 1]``: how much that label should be trusted (see
  :mod:`tools.human_optim.confidence`).
* an external **image model** that outputs a contact probability per limb per
  frame, straight from the video.

We combine them the plain way: add their log-odds. A confidence ``c`` on a label
is read as the probability ``p_geo = 0.5 + 0.5·c`` that the limb is touching (or
``0.5 - 0.5·c`` when the label says free), so a zero-confidence label — a joint
the reconstruction knows nothing about — enters the sum as ``logit(0.5) = 0`` and
lets the model decide alone. The model's probability is clipped into
``[pred_clamp, 1-pred_clamp]`` first, which caps its say at ``±logit(1-clamp)``.
That clip is the *only* guard against the model's false positives — there are no
gates, thresholds or special cases:

    logit(fused) = logit(p_geo) + logit(p_pred)

The fused label is the sign of that sum, and its confidence is
``|2·sigmoid(sum) - 1|`` (equivalently ``|tanh(sum/2)|``) — 0 when the two
witnesses exactly cancel, near 1 when they agree strongly.

Only limb-frames where the fused label *differs* from the geometric one are
written; agreeing frames keep their original per-joint confidence untouched. A
flip that survives fewer than ``min_frames`` consecutive frames is dropped — the
stage's own span rule, applied symmetrically to both directions.

A frame with no usable prediction (the model marked it invalid, or the person is
absent from the prediction file) enters as ``p_pred = 0.5`` and can therefore
never change a label: ``logit(p_geo)`` alone always agrees in sign with the
geometric label. Exact ties keep the geometric label for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .completion import _min_span

FUSION_METHOD = "predfuse-logodds-v1"
# The model's limb order, asserted against the prediction file.
PRED_LIMBS: tuple[str, ...] = ("left_hand", "right_hand", "left_foot", "right_foot")
FUSION_SOURCE_BIT = 8      # contact_source bit for a label the fusion created

# SMPLXMid 52-joint layout: a hand limb is the wrist + its 15 finger joints, a foot
# limb is the ankle + the foot (toe) joint. Members decide the limb's geometric state
# and are what a veto clears; reps are the joints a rescue turns on (a fused contact
# says "this limb touches", not which finger).
_LIMB_MEMBERS: tuple[tuple[int, ...], ...] = (
    (20, *range(22, 37)),
    (21, *range(37, 52)),
    (7, 10),
    (8, 11),
)
_LIMB_REPS: tuple[tuple[int, ...], ...] = ((20,), (21,), (7, 10), (8, 11))
_N_JOINTS = 52
# A confidence of exactly 1 would be an infinite log-odds no prediction could ever
# move; cap it so both witnesses always stay in the sum.
_GEO_CONF_CAP = 0.98


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def fuse_person(joint_contact: np.ndarray, label_confidence: np.ndarray,
                contact_source: np.ndarray, pred_prob: np.ndarray, *,
                min_frames: int, pred_clamp: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one person's per-limb predictions into their finished per-joint labels.

    Pure: the inputs are never modified, so a diagnostic can re-run this on a saved
    (unfused) ``contacts_<tag>.npz`` to sweep ``pred_clamp``.

    :param joint_contact: ``(N, 52)`` bool — the completed geometric contact labels.
    :param label_confidence: ``(N, 52)`` float in ``[0, 1]`` — trust in each label.
    :param contact_source: ``(N, 52)`` int8 provenance bitmask (1=detector, 2=gate,
        4=force); the fusion adds bit 8.
    :param pred_prob: ``(N, 4)`` model contact probability per :data:`PRED_LIMBS`,
        ``NaN`` where the model has no usable prediction for that frame.
    :param min_frames: drop a flip that lasts fewer than this many frames.
    :param pred_clamp: clip the model probability into ``[pred_clamp, 1-pred_clamp]``.
    :returns: ``(joint_contact, label_confidence, contact_source, changed (N,4) int8)``
        with ``changed`` = +1 where the fusion turned a limb on, -1 where it turned
        one off, 0 elsewhere.
    """
    jc = np.array(joint_contact, dtype=bool, copy=True)
    conf = np.array(label_confidence, dtype=np.float32, copy=True)
    src = np.array(contact_source, dtype=np.int8, copy=True)
    prob = np.asarray(pred_prob, dtype=np.float64)
    if jc.ndim != 2 or jc.shape[1] != _N_JOINTS:
        raise ValueError(f"joint_contact must have shape (N,{_N_JOINTS}), got {jc.shape}")
    n_frames = jc.shape[0]
    if conf.shape != jc.shape or src.shape != jc.shape:
        raise ValueError("label_confidence and contact_source must match joint_contact's shape")
    if prob.shape != (n_frames, len(PRED_LIMBS)):
        raise ValueError(f"pred_prob must have shape {(n_frames, len(PRED_LIMBS))}, got {prob.shape}")
    if not 0.0 < pred_clamp < 0.5:
        raise ValueError(f"pred_clamp must lie in (0, 0.5), got {pred_clamp}")

    # Geometric limb state and its confidence, folded the way the viewer folds a hand:
    # a contact limb takes its strongest touching member, a free limb the weakest of
    # its representative joints (free is an AND, and a hand's fingers would dominate).
    limb_on = np.zeros((n_frames, len(PRED_LIMBS)), dtype=bool)
    geo_conf = np.zeros((n_frames, len(PRED_LIMBS)), dtype=np.float64)
    for c, (members, reps) in enumerate(zip(_LIMB_MEMBERS, _LIMB_REPS)):
        member_on = jc[:, list(members)]
        member_conf = conf[:, list(members)].astype(np.float64)
        on = member_on.any(axis=1)
        touching = np.where(member_on, member_conf, -np.inf).max(axis=1)
        free = conf[:, list(reps)].astype(np.float64).min(axis=1)
        limb_on[:, c] = on
        geo_conf[:, c] = np.where(on, touching, free)

    c_geo = np.clip(geo_conf, 0.0, _GEO_CONF_CAP)
    p_geo = np.where(limb_on, 0.5 + 0.5 * c_geo, 0.5 - 0.5 * c_geo)
    p_pred = np.where(np.isfinite(prob), np.clip(prob, pred_clamp, 1.0 - pred_clamp), 0.5)
    logit_fused = _logit(p_geo) + _logit(p_pred)
    fused = np.where(logit_fused == 0.0, limb_on, logit_fused > 0.0)
    fused_conf = np.abs(np.tanh(0.5 * logit_fused)).astype(np.float32)

    rescue = _min_span(fused & ~limb_on, int(min_frames))
    veto = _min_span(limb_on & ~fused, int(min_frames))
    for c, (members, reps) in enumerate(zip(_LIMB_MEMBERS, _LIMB_REPS)):
        rows = np.flatnonzero(rescue[:, c])
        for j in reps:
            jc[rows, j] = True
            src[rows, j] |= np.int8(FUSION_SOURCE_BIT)
            conf[rows, j] = fused_conf[rows, c]
        rows = np.flatnonzero(veto[:, c])
        for j in members:            # clearing every member keeps joint_contact == (source != 0)
            jc[rows, j] = False
            src[rows, j] = 0
            conf[rows, j] = fused_conf[rows, c]
    return jc, conf, src, rescue.astype(np.int8) - veto.astype(np.int8)


def load_pred_prob_limb(predictions_dir: str | Path, object_ids: np.ndarray,
                        frame_indices: np.ndarray) -> np.ndarray | None:
    """Load ``predictions/contacts.npz`` aligned to the pipeline's people and frames.

    The prediction file indexes **video** frames; the pipeline's rows are the
    vggt-selected subset listed in ``geometry/transform.npz``'s ``frame_indices``,
    so rows are matched by video frame index, and people by ``object_ids``.

    :param predictions_dir: the clip's ``out/<stem>/`` directory.
    :param object_ids: ``(P,)`` the pipeline's tracked object ids.
    :param frame_indices: ``(N,)`` video frame index of each pipeline row.
    :returns: ``(P, N, 4)`` float32 probability per :data:`PRED_LIMBS`, ``NaN`` where
        the model has no usable prediction, or ``None`` when the file is absent.
    """
    path = Path(predictions_dir) / "predictions" / "contacts.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as pf:
        probs = np.asarray(pf["probs"], np.float32)
        pred_valid = np.asarray(pf["valid_mask"], bool)
        pred_oids = [int(v) for v in np.asarray(pf["object_ids"])]
        pred_frames = np.asarray(pf["frame_indices"], np.int64)
        limbs = tuple(str(v) for v in np.asarray(pf["limbs"]))
    if limbs != PRED_LIMBS:
        raise ValueError(f"{path}: limb order {limbs} != expected {PRED_LIMBS}")
    want = np.asarray(frame_indices, np.int64)
    order = np.argsort(pred_frames)
    slot = np.clip(np.searchsorted(pred_frames[order], want), 0, pred_frames.size - 1)
    row = order[slot]
    hit = pred_frames[row] == want
    out = np.full((len(object_ids), want.size, len(PRED_LIMBS)), np.nan, np.float32)
    for p, oid in enumerate(np.asarray(object_ids, np.int64)):
        if int(oid) not in pred_oids:
            continue
        q = pred_oids.index(int(oid))
        take = hit & pred_valid[q][row]
        out[p, take] = probs[q][row[take]]
    return out
