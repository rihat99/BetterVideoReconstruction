"""Rewrite the label confidence inside existing ``contacts_<tag>.npz`` artifacts.

The confidence is a function of things the artifact already stores — the folded label,
the drift, the valid mask and the frame rate — so a new fit does not need the stage to
run again. This script recomputes ``joint_label_confidence`` and
``frame_label_confidence`` in place, drops the witnesses the old confidence needed, and
stamps the new method. The binary labels are never touched: the joint fold is
recomputed and checked against the saved one.

    uv run --no-sync python scripts/recompute_contact_confidence.py \\
        --predictions out/ --contacts 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.config import load_toml  # noqa: E402
from tools.human_optim.confidence import (  # noqa: E402
    CONTACT_CONFIDENCE_METHOD, confidence_coefficients, fold_contacts, fold_drift,
    frame_report_joints, label_confidence,
)
from tools.npz_io import atomic_savez  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"

#: Fields the retired source-conditional confidence wrote and nothing reads any more.
_RETIRED = (
    "frame_visibility", "frame_image_visibility", "frame_occlusion_visibility",
    "frame_occlusion_gap_cm", "frame_surface_evidence", "frame_motion_evidence",
    "frame_depth_trust", "frame_depth_points", "confidence_knobs", "depth_fit_knobs",
)


def find_artifacts(roots: list[Path], tag: str) -> list[Path]:
    """Every ``contacts_<tag>.npz`` under the given paths.

    A path may be a scene dir (``out/<scene>/``), a corpus feature dir holding the npz
    directly, or any parent of either.
    """
    name = f"contacts_{tag}.npz"
    found: list[Path] = []
    for root in roots:
        if (root / "human_optim" / name).exists():
            found.append(root / "human_optim" / name)
        elif (root / name).exists():
            found.append(root / name)
        else:
            found += sorted(root.rglob(name))
    return found


def recompute(path: Path, coeffs: dict) -> None:
    """Recompute one artifact's confidence in place."""
    saved = dict(np.load(path, allow_pickle=True))
    joint_names = [str(n) for n in saved["joint_names"]]
    parents = np.asarray(saved["parents"], dtype=np.int32)
    report_joint = frame_report_joints(saved["frame_parent_joint"], joint_names, parents)

    frame_contact = np.asarray(saved["frame_contact"], dtype=bool)
    joint_contact = fold_contacts(frame_contact, report_joint, len(parents))
    if "joint_contact" in saved and not np.array_equal(
            joint_contact, np.asarray(saved["joint_contact"], dtype=bool)):
        raise ValueError("the OR fold does not reproduce the saved joint_contact")
    joint_drift = fold_drift(np.asarray(saved["frame_drift_m"]), report_joint, len(parents))
    valid = np.asarray(saved["valid_mask"], dtype=bool)
    fps = float(saved["fps"])

    confidence = np.stack([
        label_confidence(joint_contact[p], joint_drift[p], valid[p], fps, joint_names, coeffs)
        for p in range(joint_contact.shape[0])])

    for key in _RETIRED:
        saved.pop(key, None)
    saved["joint_contact"] = joint_contact
    saved["joint_label_confidence"] = confidence
    saved["frame_label_confidence"] = confidence[..., report_joint]
    saved["frame_report_joint"] = report_joint
    saved["confidence_method"] = np.str_(CONTACT_CONFIDENCE_METHOD)
    saved["confidence_motion_source"] = np.str_("drift_pre")
    saved["confidence_coefficients"] = np.str_(json.dumps(coeffs))
    atomic_savez(path, **saved)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", type=Path, nargs="+", required=True,
                    help="scene dirs, corpus feature dirs, or any parent of them")
    ap.add_argument("--contacts", default="1", choices=("1", "2"), help="contacts_<tag>.npz")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true",
                    help="rewrite artifacts that already carry the current method")
    args = ap.parse_args(argv)

    coeffs = confidence_coefficients(load_toml(args.config)["contact_confidence"])
    artifacts = find_artifacts(args.predictions, args.contacts)
    done = skipped = failed = 0
    for path in artifacts:
        try:
            with np.load(path, allow_pickle=True) as saved:
                method = str(saved["confidence_method"]) if "confidence_method" in saved else ""
            if method == CONTACT_CONFIDENCE_METHOD and not args.force:
                skipped += 1
                print(f"[skip] {path}  (already {method})")
                continue
            recompute(path, coeffs)
        except Exception as error:          # a broken artifact must not kill the sweep
            failed += 1
            print(f"[fail] {path}  {type(error).__name__}: {error}")
            continue
        done += 1
        print(f"[done] {path}")
    print(f"\n{done} rewritten, {skipped} already current, {failed} failed "
          f"({len(artifacts)} artifacts found)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
