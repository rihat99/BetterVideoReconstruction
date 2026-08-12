"""Visibility weighting of the kinematics_1 visual terms + the held-frame smoothness gate.

Two properties are load-bearing and easy to break:

* **Denominator invariance** — every weighted term divides by the *unweighted* count, so
  lowering one joint's weight lowers the term instead of pushing its share onto the other
  joints. Each test therefore checks the exact arithmetic, not just "the number moved".
* **The chain fallback** — joints Sapiens has no keypoint for inherit their parent's
  visibility, so a foot follows its ankle and the fingers follow their wrist.
"""

from __future__ import annotations

import torch

from tools.human_optim.keypoints import Correspondence, visibility_weights
from tools.human_optim.losses import (
    init_reprojection_gm, pose_prior_loss, sapiens_reprojection_gm)
from tools.human_optim.optimizer import _valid_diff_rows

# A toy body: root -> shoulder -> elbow -> hand, plus a torso joint no keypoint watches.
_PARENTS = [-1, 0, 1, 2, 0]
_SHOULDER, _ELBOW, _HAND, _TORSO = 1, 2, 3, 4
_NUM_KEYPOINTS = 8


def _corr(num_landmarks: int = 0) -> Correspondence:
    """Shoulder and elbow are corresponded (keypoints 3 and 5); the rest are not."""
    long = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
    sap = [3, 5] + list(range(num_landmarks))
    return Correspondence(joint_model=long([_SHOULDER, _ELBOW]),
                          lm_frame=long(list(range(num_landmarks))), sap=long(sap))


def _scores(num_frames: int = 12) -> torch.Tensor:
    """Scores flat in time so the 90th percentile equals the level itself."""
    scores = torch.full((num_frames, _NUM_KEYPOINTS), 0.9)
    scores[:, 3] = 0.6            # shoulder: a badly *calibrated* keypoint, never occluded
    scores[:, 5] = 0.9
    scores[num_frames // 2:, 5] = 0.3   # elbow: a real drop halfway through = occlusion
    return scores


def test_visibility_is_relative_to_the_keypoints_own_clear_level():
    """A uniformly low-scoring keypoint reads fully visible; a keypoint that DROPS does not."""
    keypoint, joint = visibility_weights(_scores(), _corr(), _PARENTS,
                                         strength=1.0, window_s=0.0, fps=30.0)
    assert torch.allclose(keypoint[:, 3], torch.full((12,), 1.0))          # 0.6 / 0.6
    assert torch.allclose(joint[:6, _ELBOW], torch.full((6,), 1.0))        # 0.9 / 0.9
    assert torch.allclose(joint[6:, _ELBOW], torch.full((6,), 1.0 / 3.0))  # 0.3 / 0.9


def test_visibility_strength_scales_the_shortfall_and_zero_is_off():
    """Strength multiplies the DROP, so a clear keypoint barely moves and 0 is exactly off."""
    _, joint = visibility_weights(_scores(), _corr(), _PARENTS,
                                  strength=1.2, window_s=0.0, fps=30.0)
    assert torch.allclose(joint[:6, _ELBOW], torch.ones(6))                 # ratio 1 stays 1
    assert torch.allclose(joint[6:, _ELBOW], torch.full((6,), 0.2), atol=1e-6)   # 1 - 1.2*(2/3)
    _, strong = visibility_weights(_scores(), _corr(), _PARENTS,
                                   strength=3.0, window_s=0.0, fps=30.0)
    assert float(strong[:, _ELBOW].min()) == 0.0                            # clamped, never negative
    assert torch.allclose(strong[:, _SHOULDER], torch.ones(12))             # a clear joint is untouched
    keypoint, joint = visibility_weights(_scores(), _corr(), _PARENTS,
                                         strength=0.0, window_s=0.0, fps=30.0)
    assert torch.equal(joint, torch.ones_like(joint))
    assert torch.equal(keypoint, torch.ones_like(keypoint))


def test_uncorresponded_joints_inherit_their_parent():
    """Hand follows elbow down the chain; a torso joint hanging off the root stays at 1."""
    _, joint = visibility_weights(_scores(), _corr(), _PARENTS,
                                  strength=1.0, window_s=0.0, fps=30.0)
    assert torch.equal(joint[:, _HAND], joint[:, _ELBOW])
    assert torch.allclose(joint[:, _TORSO], torch.ones(12))
    assert torch.allclose(joint[:, 0], torch.ones(12))


def test_untracked_frames_keep_full_weight():
    """A held frame has no observation, so its priors must not be loosened."""
    frame_valid = torch.ones(12, dtype=torch.bool)
    frame_valid[8:] = False
    keypoint, joint = visibility_weights(_scores(), _corr(), _PARENTS, strength=1.0,
                                         window_s=0.0, fps=30.0, frame_valid=frame_valid)
    assert torch.allclose(joint[8:], torch.ones_like(joint[8:]))
    assert torch.allclose(keypoint[8:], torch.ones_like(keypoint[8:]))
    assert torch.allclose(joint[6:8, _ELBOW], torch.full((2,), 1.0 / 3.0))


def _pose_prior_inputs(num_frames: int = 4, num_joints: int = 5):
    torch.manual_seed(0)
    q = torch.randn(num_frames, 7 + 4 * (num_joints - 1))
    q_init = torch.randn(num_frames, 7 + 4 * (num_joints - 1))
    quats = q[:, 7:].reshape(num_frames, num_joints - 1, 4)
    q[:, 7:] = (quats / quats.norm(dim=-1, keepdim=True)).reshape(num_frames, -1)
    quats = q_init[:, 7:].reshape(num_frames, num_joints - 1, 4)
    q_init[:, 7:] = (quats / quats.norm(dim=-1, keepdim=True)).reshape(num_frames, -1)
    return q, q_init, num_joints


def test_pose_prior_weight_of_one_is_the_unweighted_loss():
    q, q_init, num_joints = _pose_prior_inputs()
    plain = pose_prior_loss(q, q_init, num_joints)
    ones = pose_prior_loss(q, q_init, num_joints,
                           joint_weight=torch.ones(q.shape[0], num_joints - 1))
    assert torch.allclose(plain, ones, atol=1e-6)


def test_pose_prior_holds_its_denominator():
    """Zeroing one joint's weight removes exactly that joint's share — nothing rebalances."""
    q, q_init, num_joints = _pose_prior_inputs()
    weight = torch.ones(q.shape[0], num_joints - 1)
    weight[:, 2] = 0.0
    quats = q[:, 7:].reshape(q.shape[0], num_joints - 1, 4)
    init_quats = q_init[:, 7:].reshape(q.shape[0], num_joints - 1, 4)
    share = (1.0 - (quats[:, 2] * init_quats[:, 2]).sum(-1).square()).sum() / quats[:, :, 0].numel()
    dropped = pose_prior_loss(q, q_init, num_joints, joint_weight=weight)
    assert torch.allclose(pose_prior_loss(q, q_init, num_joints) - share, dropped, atol=1e-6)


def test_pose_prior_weight_composes_with_num_prior_joints():
    q, q_init, num_joints = _pose_prior_inputs()
    weight = torch.full((q.shape[0], num_joints - 1), 0.5)
    half = pose_prior_loss(q, q_init, num_joints, num_prior_joints=2, joint_weight=weight)
    full = pose_prior_loss(q, q_init, num_joints, num_prior_joints=2)
    assert torch.allclose(0.5 * full, half, atol=1e-6)


def _reproj_inputs(num_frames: int = 3, num_joints: int = 4):
    torch.manual_seed(1)
    joints = torch.randn(num_frames, num_joints, 3) * 0.2 + torch.tensor([0.0, 0.0, 3.0])
    extrinsics = torch.eye(4).expand(num_frames, 4, 4).contiguous()
    intrinsics = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics = intrinsics.expand(num_frames, 3, 3).contiguous()
    target = torch.randn(num_frames, num_joints, 2) * 20.0 + 300.0
    return joints, torch.arange(num_joints), extrinsics, intrinsics, target


def test_init_reprojection_weight_of_one_is_the_unweighted_loss():
    joints, body_idx, extrinsics, intrinsics, target = _reproj_inputs()
    ones = torch.ones(joints.shape[0], body_idx.numel())
    for mask in (None, torch.tensor([True, True, False])):
        plain = init_reprojection_gm(joints, body_idx, extrinsics, intrinsics, target,
                                     gm_scale_px=30.0, frame_mask=mask)
        weighted = init_reprojection_gm(joints, body_idx, extrinsics, intrinsics, target,
                                        gm_scale_px=30.0, frame_mask=mask, joint_weight=ones)
        assert torch.allclose(plain, weighted, atol=1e-6)


def test_init_reprojection_holds_its_denominator():
    joints, body_idx, extrinsics, intrinsics, target = _reproj_inputs()
    weight = torch.ones(joints.shape[0], body_idx.numel())
    weight[:, 1] = 0.0
    plain = init_reprojection_gm(joints, body_idx, extrinsics, intrinsics, target,
                                 gm_scale_px=30.0)
    dropped = init_reprojection_gm(joints, body_idx, extrinsics, intrinsics, target,
                                   gm_scale_px=30.0, joint_weight=weight)
    only_one = init_reprojection_gm(joints, body_idx[1:2], extrinsics, intrinsics,
                                    target[:, 1:2], gm_scale_px=30.0)
    share = only_one / body_idx.numel()   # its share of the FULL-width mean
    assert torch.allclose(plain - share, dropped, atol=1e-6)


def _sapiens_inputs(num_frames: int = 3):
    torch.manual_seed(2)
    joints = torch.randn(num_frames, 5, 3) * 0.2 + torch.tensor([0.0, 0.0, 3.0])
    vertices = torch.randn(num_frames, 6, 3) * 0.2 + torch.tensor([0.0, 0.0, 3.0])
    extrinsics = torch.eye(4).expand(num_frames, 4, 4).contiguous()
    intrinsics = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics = intrinsics.expand(num_frames, 3, 3).contiguous()
    target = torch.randn(num_frames, _NUM_KEYPOINTS, 2) * 20.0 + 300.0
    scores = torch.full((num_frames, _NUM_KEYPOINTS), 0.8)
    frame_vertex_ids = torch.arange(6)
    return joints, vertices, extrinsics, intrinsics, target, scores, frame_vertex_ids


def test_sapiens_reprojection_visibility_of_one_is_the_unweighted_loss():
    joints, vertices, extrinsics, intrinsics, target, scores, fvids = _sapiens_inputs()
    corr = _corr(num_landmarks=2)
    plain = sapiens_reprojection_gm(joints, vertices, extrinsics, intrinsics, target, scores,
                                    corr, fvids, score_thresh=0.3, gm_scale_px=40.0)
    weighted = sapiens_reprojection_gm(
        joints, vertices, extrinsics, intrinsics, target, scores, corr, fvids,
        score_thresh=0.3, gm_scale_px=40.0, visibility=torch.ones_like(scores))
    assert torch.allclose(plain, weighted, atol=1e-6)


def test_sapiens_reprojection_holds_its_score_denominator():
    """Halving every visibility halves the term — it does not renormalize back to plain."""
    joints, vertices, extrinsics, intrinsics, target, scores, fvids = _sapiens_inputs()
    corr = _corr(num_landmarks=2)
    plain = sapiens_reprojection_gm(joints, vertices, extrinsics, intrinsics, target, scores,
                                    corr, fvids, score_thresh=0.3, gm_scale_px=40.0)
    half = sapiens_reprojection_gm(
        joints, vertices, extrinsics, intrinsics, target, scores, corr, fvids,
        score_thresh=0.3, gm_scale_px=40.0, visibility=torch.full_like(scores, 0.5))
    assert torch.allclose(0.5 * plain, half, atol=1e-6)


def test_valid_diff_rows_drops_only_the_stencils_touching_a_gap():
    frame_valid = torch.ones(10, dtype=torch.bool)
    frame_valid[4] = False
    assert torch.equal(_valid_diff_rows(frame_valid, 1, 10),
                       torch.tensor([True, True, True, False, False, True, True, True, True]))
    accel = _valid_diff_rows(frame_valid, 2, 10)
    assert accel.shape == (8,) and int(accel.sum()) == 5
    assert not bool(accel[2:5].any()) and bool(accel[:2].all()) and bool(accel[5:].all())
    # snap: 6 rows of 5 frames each; the gap at frame 4 sits inside rows 0..4, so 1 survives.
    assert torch.equal(_valid_diff_rows(frame_valid, 4, 10),
                       torch.tensor([False, False, False, False, False, True]))


def test_valid_diff_rows_is_none_when_there_is_nothing_to_gate():
    assert _valid_diff_rows(None, 2, 10) is None
    assert _valid_diff_rows(torch.ones(10, dtype=torch.bool), 2, 10) is None   # all valid
    assert _valid_diff_rows(torch.ones(3, dtype=torch.bool), 4, 3) is None     # clip too short
