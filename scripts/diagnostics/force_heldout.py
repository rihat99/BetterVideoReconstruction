"""Score the frozen force recipe on held-out ClimbingVideos scenes.

Every knob was chosen on the 13 local scenes. These scenes come from the annotated
ClimbingVideos DB, filtered to source videos that appear nowhere in that tuning set,
and the recipe was frozen in code before any of them was scored. The DB is never
written: each scene is a local farm of read-only symlinks with ``kinematic_1`` and the
contact readouts produced locally.

Two arms, both from the shipped ``estimate_contacts`` stage on the same body: the
geometric detector alone (the force threshold raised out of reach) and the shipped
force-union readout.

Six of the eleven scenes share one source video, so a plain pool over scenes would
mostly measure that one video. The per-source macro average is reported alongside.

Usage::

    uv run --no-sync python scripts/diagnostics/force_heldout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from score_frame_contacts import _GROUPS, _pool, metrics  # noqa: E402

import force_recovery  # noqa: E402

_FARM = _REPO_ROOT / "out" / "_expK_heldout"
_ANN = _REPO_ROOT / "scratch" / "force_recovery" / "heldout_data"
_ARMS = {"gates": "gates_only", "force|gates": "v2_nooverlay"}


def _source(stem: str) -> str:
    """The source video a scene was cut from — everything before the trailing ``_NNNN``."""
    return stem.rsplit("_", 1)[0]


def main() -> int:
    force_recovery._ANNOTATIONS[0] = _ANN          # score against the held-out annotations
    stems = sorted(p.name for p in _FARM.iterdir() if p.is_dir())

    per_scene: dict[str, dict[str, dict]] = {}
    failed: list[str] = []
    for stem in stems:
        arms = {}
        for arm, tag in _ARMS.items():
            path = _FARM / stem / "human_optim" / f"contacts_{tag}.npz"
            if not path.exists():
                break
            npz = np.load(path, allow_pickle=True)
            meta = dict(frame_names=[str(n) for n in npz["contact_frame_names"]],
                        valid_mask=np.asarray(npz["valid_mask"], bool),
                        object_ids=[int(o) for o in npz["object_ids"]])
            labels = list(np.asarray(npz["frame_contact"], bool))
            arms[arm] = force_recovery.score_labels(stem, labels, meta)
        if len(arms) != len(_ARMS):
            failed.append(stem)
            continue
        per_scene[stem] = arms

    header = (f"{'scene':22s} {'source':16s} {'arm':12s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
              f"{'prec':>7s} {'rec':>7s} {'F1':>7s}")
    print("HELD-OUT — never-tuned ClimbingVideos scenes, frozen recipe, no re-tuning")
    print("=" * len(header))
    print(header)
    show = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
    for stem, arms in per_scene.items():
        for arm, joints in arms.items():
            m = metrics(_pool(list(joints.values())))
            print(f"{stem:22s} {_source(stem):16s} {arm:12s} {m['tp']:6d} {m['fp']:6d} "
                  f"{m['fn']:6d} {show(m['precision'])}{show(m['recall'])}{show(m['f1'])}")

    knees = set(_GROUPS["knees"])
    print("-" * len(header))
    for arm in _ARMS:
        rows = [per_scene[s][arm] for s in per_scene]
        m = metrics(_pool([c for j in rows for c in j.values()]))
        nk = metrics(_pool([c for j in rows for n, c in j.items() if n not in knees]))
        print(f"{'POOLED (all scenes)':38s} {arm:12s} {m['tp']:6d} {m['fp']:6d} {m['fn']:6d} "
              f"{show(m['precision'])}{show(m['recall'])}{show(m['f1'])}"
              f"   no-knee F1 {show(nk['f1'])}")

    print("-" * len(header))
    sources = sorted({_source(s) for s in per_scene})
    for arm in _ARMS:
        f1s = []
        for source in sources:
            rows = [per_scene[s][arm] for s in per_scene if _source(s) == source]
            m = metrics(_pool([c for j in rows for c in j.values()]))
            if m["f1"] is not None:
                f1s.append(m["f1"])
        print(f"{'MACRO over ' + str(len(sources)) + ' source videos':38s} {arm:12s} "
              f"{'':20s} mean F1 {np.mean(f1s):7.3f}")

    print("-" * len(header))
    print(f"{'per-source F1':22s} " + "  ".join(f"{s[:14]:>14s}" for s in sources))
    for arm in _ARMS:
        cells = []
        for source in sources:
            rows = [per_scene[s][arm] for s in per_scene if _source(s) == source]
            m = metrics(_pool([c for j in rows for c in j.values()]))
            cells.append("             -" if m["f1"] is None else f"{m['f1']:14.3f}")
        print(f"{arm:22s} " + "  ".join(cells))

    print(f"\nscenes scored: {len(per_scene)}/{len(stems)}")
    if failed:
        print(f"FAILED (reported, not dropped): {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
