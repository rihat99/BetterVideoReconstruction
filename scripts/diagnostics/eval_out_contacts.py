"""Score the stage's contact labels against the manual annotations on the out/ clips.

The 13 clips under ``out/`` are hand-annotated: ``data/<stem>.npz`` holds a tri-state
label per coarse joint per frame (1 contact, 0 free, -1 unknown/not annotated). This
script folds both sides to the **four limbs** the annotations and the contact
predictions agree on — left/right hand, left/right foot — and reports precision,
recall and F1 per limb, per clip and pooled over every scored limb-frame.

Folding, on the prediction side: a hand touches if its wrist or any finger joint does,
a foot limb if its ankle or its foot (toe) joint does. On the annotation side each hand
is its own row; the foot is the annotation's foot row, OR'd with its ankle row when the
clip uses the 14-joint schema (the older 10-joint schema has no ankle row — its foot row
already means ankle-or-foot). Only frames where the annotation is 0 or 1 **and** the
reconstructed body is valid are scored; annotated people marked ``ignored`` are skipped.

Two extra modes:

* ``--baseline-root`` scores a second tree the same way and prints it next to the first
  (e.g. ``out/_prefusion_baseline``, the snapshot taken before predicted-contact fusion).
  Both a full ``<scene>/human_optim/contacts_<tag>.npz`` layout and a flat
  ``<scene>/contacts_<tag>.npz`` snapshot are accepted.
* ``--sweep-clamp`` re-runs the predicted-contact fusion offline on those baseline
  labels (via the pure :func:`tools.human_optim.fusion.fuse_person`) for several values
  of the model's clamp, and reports the pooled score of each — the clamp is the only
  knob that bounds how far the image model may move a label.

Usage::

    uv run --no-sync python scripts/diagnostics/eval_out_contacts.py \\
        --baseline-root out/_prefusion_baseline --json scratch/out_contacts.json
    ... --tag 2                       # score contacts_2 instead
    ... --scenes boulder_1,sport_climb
    ... --sweep-clamp                 # pooled score per pred_clamp, from the baseline labels
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "diagnostics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from label_readout import prf  # noqa: E402
from tools.human_optim.fusion import (  # noqa: E402
    _LIMB_MEMBERS, PRED_LIMBS, fuse_person, load_pred_prob_limb)

_DEFAULT_ROOT = _REPO_ROOT / "out"
_DATA_DIR = _REPO_ROOT / "data"
_ACCEPTED_CONFIDENCE_SCHEMA = (7, 8)      # 7 = pre-fusion snapshots, 8 = with fusion
_CLAMP_SWEEP = (0.05, 0.10, 0.20, 0.35)


def contacts_path(root: Path, scene: str, tag: str) -> Path | None:
    """The scene's ``contacts_<tag>.npz`` in either the full or the flat-snapshot layout."""
    for rel in (Path("human_optim") / f"contacts_{tag}.npz", Path(f"contacts_{tag}.npz")):
        if (root / scene / rel).exists():
            return root / scene / rel
    return None


def scenes_in(root: Path, tag: str) -> list[str]:
    """Scene directories under ``root`` that hold a contacts npz and an annotation."""
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and contacts_path(root, d.name, tag) is not None
        and (_DATA_DIR / f"{d.name}.npz").exists())


def fold_pred_limbs(joint_contact: np.ndarray) -> np.ndarray:
    """``(P,N,52)`` per-joint contact → ``(P,N,4)`` per-limb contact (OR over members)."""
    return np.stack([joint_contact[..., list(members)].any(-1)
                     for members in _LIMB_MEMBERS], axis=-1)


def _tri_or(rows: list[np.ndarray]) -> np.ndarray:
    """Tri-state OR of annotation rows: 1 if any says contact, else 0 if any says free."""
    stack = np.stack(rows, 0)
    return np.where((stack == 1).any(0), np.int8(1),
                    np.where((stack == 0).any(0), np.int8(0), np.int8(-1)))


def load_gt_limbs(stem: str, n_frames: int) -> tuple[np.ndarray, list[int]]:
    """Manual annotation → ``(P_ann, N, 4)`` tri-state limb labels + its object ids.

    ``ignored`` people are dropped. The annotation stores frames on the LAST axis.
    """
    with np.load(_DATA_DIR / f"{stem}.npz", allow_pickle=True) as ann:
        contacts = np.asarray(ann["contacts"], np.int8)              # (P, J, N)
        names = [str(n) for n in np.asarray(ann["joint_names"])]
        oids = [int(v) for v in np.asarray(ann["object_ids"])]
        ignored = np.asarray(ann["ignored"], bool)
    if contacts.shape[-1] < n_frames:
        raise ValueError(f"{stem}: annotation has {contacts.shape[-1]} frames < body {n_frames}")
    keep = [i for i in range(len(oids)) if not ignored[i]]
    limbs = []
    for i in keep:
        rows = []
        for limb in PRED_LIMBS:
            group = [limb] + ([f"{limb.split('_')[0]}_ankle"] if limb.endswith("foot") else [])
            present = [contacts[i, names.index(name), :n_frames] for name in group if name in names]
            if not present:
                raise ValueError(f"{stem}: annotation has no row for {limb}")
            rows.append(_tri_or(present))
        limbs.append(np.stack(rows, -1))
    return np.stack(limbs, 0) if limbs else np.zeros((0, n_frames, 4), np.int8), [oids[i] for i in keep]


def load_contacts(path: Path) -> dict:
    """Read the per-joint labels + confidence + provenance the contact stage wrote."""
    with np.load(path, allow_pickle=True) as cnt:
        schema = int(np.asarray(cnt["confidence_schema"]).item())
        if schema not in _ACCEPTED_CONFIDENCE_SCHEMA:
            raise ValueError(f"{path}: confidence_schema {schema} not in "
                             f"{_ACCEPTED_CONFIDENCE_SCHEMA}; regenerate estimate_contacts")
        return dict(
            joint_contact=np.asarray(cnt["joint_contact"], bool),
            label_confidence=np.asarray(cnt["label_confidence"], np.float32),
            contact_source=np.asarray(cnt["contact_source"], np.int8),
            valid_mask=np.asarray(cnt["valid_mask"], bool),
            object_ids=[int(v) for v in np.asarray(cnt["object_ids"])],
            fusion_method=str(np.asarray(cnt["fusion_method"]).item())
            if "fusion_method" in cnt.files else "none",
            schema=schema,
        )


def score_pairs(pred_limbs: np.ndarray, valid: np.ndarray, gt: np.ndarray,
                pred_oids: list[int], gt_oids: list[int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-limb ``(pred, gt)`` boolean pairs over the scored limb-frames of one scene."""
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c, limb in enumerate(PRED_LIMBS):
        pr, gg = [], []
        for p, oid in enumerate(pred_oids):
            if oid not in gt_oids:
                continue
            a = gt_oids.index(oid)
            scored = valid[p] & (gt[a, :, c] >= 0)
            pr.append(pred_limbs[p, scored, c])
            gg.append(gt[a, scored, c] == 1)
        pairs[limb] = (np.concatenate(pr) if pr else np.zeros(0, bool),
                       np.concatenate(gg) if gg else np.zeros(0, bool))
    return pairs


def _pooled(pairs_list: list[dict]) -> dict[str, dict]:
    """Pool per-limb pairs over scenes → ``{limb: {p,r,f1,...}}`` plus an ``ALL`` row."""
    out: dict[str, dict] = {}
    every_pred, every_gt = [], []
    for limb in PRED_LIMBS:
        pr = np.concatenate([p[limb][0] for p in pairs_list]) if pairs_list else np.zeros(0, bool)
        gt = np.concatenate([p[limb][1] for p in pairs_list]) if pairs_list else np.zeros(0, bool)
        every_pred.append(pr)
        every_gt.append(gt)
        out[limb] = _row(pr, gt)
    out["ALL"] = _row(np.concatenate(every_pred), np.concatenate(every_gt))
    return out


def _row(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, r, f1 = prf(pred, gt)
    return dict(p=float(p), r=float(r), f1=float(f1), n_scored=int(pred.size),
                n_pred=int(pred.sum()), n_gt=int(gt.sum()))


def _print_table(title: str, per_scene: dict[str, dict], pooled: dict[str, dict]) -> None:
    head = "".join(f"{limb:>17s}" for limb in (*PRED_LIMBS, "ALL"))
    print(f"\n== {title}")
    print(f"{'scene':24s}{head}")
    print(f"{'':24s}" + "".join(f"{'P    R   F1':>17s}" for _ in range(5)))
    for scene, rows in per_scene.items():
        cells = "".join(f"{rows[k]['p']:>7.3f}{rows[k]['r']:6.3f}{rows[k]['f1']:6.3f}"
                        for k in (*PRED_LIMBS, "ALL"))
        print(f"{scene:24s}{cells}")
    cells = "".join(f"{pooled[k]['p']:>7.3f}{pooled[k]['r']:6.3f}{pooled[k]['f1']:6.3f}"
                    for k in (*PRED_LIMBS, "ALL"))
    print(f"{'POOLED':24s}{cells}")
    print(f"{'':24s}" + "".join(f"{'n=' + str(pooled[k]['n_scored']):>17s}"
                                for k in (*PRED_LIMBS, "ALL")))


def evaluate(root: Path, scenes: list[str], tag: str) -> tuple[dict, dict, list[dict]]:
    """Score one tree → ``(per-scene rows, pooled rows, the per-scene limb pairs)``."""
    per_scene: dict[str, dict] = {}
    pairs_list: list[dict] = []
    for scene in scenes:
        path = contacts_path(root, scene, tag)
        if path is None:
            print(f"[eval_out_contacts] {scene}: no contacts_{tag}.npz under {root}")
            continue
        try:
            cnt = load_contacts(path)
            n_frames = cnt["joint_contact"].shape[1]
            gt, gt_oids = load_gt_limbs(scene, n_frames)
            pairs = score_pairs(fold_pred_limbs(cnt["joint_contact"]), cnt["valid_mask"],
                                gt, cnt["object_ids"], gt_oids)
        except Exception as exc:  # noqa: BLE001 — one bad scene must not kill the sweep
            print(f"[eval_out_contacts] {scene}: FAILED {type(exc).__name__}: {exc}")
            continue
        pairs_list.append(pairs)
        per_scene[scene] = {limb: _row(*pairs[limb]) for limb in PRED_LIMBS}
        per_scene[scene]["ALL"] = _row(
            np.concatenate([pairs[limb][0] for limb in PRED_LIMBS]),
            np.concatenate([pairs[limb][1] for limb in PRED_LIMBS]))
        per_scene[scene]["fusion_method"] = cnt["fusion_method"]
    return per_scene, _pooled(pairs_list), pairs_list


def sweep_clamp(root: Path, baseline_root: Path, scenes: list[str], tag: str,
                min_frames: int) -> dict[str, dict]:
    """Pooled score of the offline fusion on the baseline labels, per ``pred_clamp``."""
    results: dict[str, dict] = {}
    loaded = []
    for scene in scenes:
        path = contacts_path(baseline_root, scene, tag)
        transform = root / scene / "geometry" / "transform.npz"
        if path is None or not transform.exists():
            print(f"[eval_out_contacts] sweep: skipping {scene} (no baseline npz or transform)")
            continue
        try:
            cnt = load_contacts(path)
            if cnt["fusion_method"] != "none":
                print(f"[eval_out_contacts] sweep: {scene} baseline is already fused "
                      f"({cnt['fusion_method']}) — re-fusing it would double-count")
                continue
            with np.load(transform, allow_pickle=True) as tr:
                frame_indices = np.asarray(tr["frame_indices"])
            prob = load_pred_prob_limb(root / scene, cnt["object_ids"], frame_indices)
            if prob is None:
                print(f"[eval_out_contacts] sweep: {scene} has no predictions/contacts.npz")
                continue
            prob[~cnt["valid_mask"]] = np.nan
            gt, gt_oids = load_gt_limbs(scene, cnt["joint_contact"].shape[1])
        except Exception as exc:  # noqa: BLE001
            print(f"[eval_out_contacts] sweep: {scene} FAILED {type(exc).__name__}: {exc}")
            continue
        loaded.append((scene, cnt, prob, gt, gt_oids))

    for clamp in _CLAMP_SWEEP:
        pairs_list, flips = [], np.zeros(2, np.int64)
        for scene, cnt, prob, gt, gt_oids in loaded:
            fused = cnt["joint_contact"].copy()
            for p in range(fused.shape[0]):
                jc, _, _, changed = fuse_person(
                    cnt["joint_contact"][p], cnt["label_confidence"][p],
                    cnt["contact_source"][p], prob[p],
                    min_frames=min_frames, pred_clamp=clamp)
                fused[p] = jc
                flips += [int((changed > 0).sum()), int((changed < 0).sum())]
            pairs_list.append(score_pairs(fold_pred_limbs(fused), cnt["valid_mask"],
                                          gt, cnt["object_ids"], gt_oids))
        results[f"{clamp:.2f}"] = dict(pooled=_pooled(pairs_list),
                                       rescued=int(flips[0]), vetoed=int(flips[1]))
    baseline = _pooled([score_pairs(fold_pred_limbs(cnt["joint_contact"]), cnt["valid_mask"],
                                    gt, cnt["object_ids"], gt_oids)
                        for _, cnt, _, gt, gt_oids in loaded])
    results["baseline"] = dict(pooled=baseline, rescued=0, vetoed=0)

    print(f"\n== pred_clamp sweep (offline fusion on {len(loaded)} baseline scenes, "
          f"min_frames={min_frames})")
    print(f"{'clamp':>8s}{'P':>8s}{'R':>8s}{'F1':>8s}{'rescued':>10s}{'vetoed':>8s}")
    for key in ("baseline", *(f"{c:.2f}" for c in _CLAMP_SWEEP)):
        row = results[key]["pooled"]["ALL"]
        print(f"{key:>8s}{row['p']:8.3f}{row['r']:8.3f}{row['f1']:8.3f}"
              f"{results[key]['rescued']:10d}{results[key]['vetoed']:8d}")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT, help="tree to score (default out/)")
    ap.add_argument("--baseline-root", type=Path, default=None,
                    help="second tree to score for comparison (e.g. out/_prefusion_baseline)")
    ap.add_argument("--tag", default="1", choices=["1", "2"], help="contacts_<tag>.npz")
    ap.add_argument("--scenes", default=None, help="comma-separated scene names (default: all)")
    ap.add_argument("--sweep-clamp", action="store_true",
                    help="pooled score of the offline fusion per pred_clamp, from the baseline tree")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="span rule for the offline fusion sweep ([contacts].min_frames)")
    ap.add_argument("--json", type=Path, default=None, help="write the full result here")
    args = ap.parse_args(argv)

    scenes = ([s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes
              else scenes_in(args.root, args.tag))
    out: dict = dict(root=str(args.root), tag=args.tag, scenes=scenes)
    per_scene, pooled, _ = evaluate(args.root, scenes, args.tag)
    _print_table(f"{args.root} — contacts_{args.tag}", per_scene, pooled)
    out["current"] = dict(per_scene=per_scene, pooled=pooled)

    if args.baseline_root is not None:
        base_scene, base_pooled, _ = evaluate(args.baseline_root, scenes, args.tag)
        _print_table(f"{args.baseline_root} — contacts_{args.tag}", base_scene, base_pooled)
        out["baseline"] = dict(root=str(args.baseline_root), per_scene=base_scene,
                               pooled=base_pooled)
        print("\n== pooled ALL delta (current - baseline)")
        cur, base = pooled["ALL"], base_pooled["ALL"]
        print(f"  P {base['p']:.3f} -> {cur['p']:.3f} ({cur['p'] - base['p']:+.3f})   "
              f"R {base['r']:.3f} -> {cur['r']:.3f} ({cur['r'] - base['r']:+.3f})   "
              f"F1 {base['f1']:.3f} -> {cur['f1']:.3f} ({cur['f1'] - base['f1']:+.3f})")

    if args.sweep_clamp:
        source = args.baseline_root if args.baseline_root is not None else args.root
        out["sweep"] = sweep_clamp(args.root, source, scenes, args.tag, args.min_frames)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\n[eval_out_contacts] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
