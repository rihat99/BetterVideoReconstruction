"""Read-only diagnostics for the scene-contact stage (``estimate_contacts``).

Reads the refined body (``human_optim/kinematic_1.npz``), its per-joint contact
npz (``human_optim/contacts_1.npz``), and the fused scene cloud
(``geometry/scene.npz``) — writes PNGs + prints summaries, changing no pipeline
state. Two diagnostics, per tracked person:

  1. **Distance histograms per joint class** (``hands`` / ``feet`` / ``other``) of
     the raw per-joint min perpendicular signed distance
     (``contacts_1.npz['joint_sdf_cm']``, pooled over the class's joints × valid
     frames). A clean contact detector wants each class **two-lobed** — a touching
     lobe near/below 0 and a free lobe well positive, split by a clear valley near
     the ``close_thresh_m`` line. A single lobe means the threshold is cutting
     through a continuum (contact there is not trustworthy).
  2. **Carve-hole probe.** For every frame a wrist moves *slowly* (< 0.2 m/s,
     central-difference of its posed world position) and is labelled *not in
     contact*, count the conf-gated scene points within a 10 cm ball of the wrist.
     The fraction of such frames with < 20 nearby points measures how often holds
     are carved out of the scene cloud (dilate-8 human removal), so contact simply
     cannot be detected there.

    .venv/bin/python scripts/diagnose_contacts.py --predictions out/boulder_1/

Writes ``out/<stem>/human_optim/diagnostics/contacts_hist_person<oid>.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETTER_HUMAN_SRC = _REPO_ROOT.parent / "better_human" / "src"
for _p in (str(_BETTER_HUMAN_SRC), str(_REPO_ROOT)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import tomllib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.human_optim import paths  # noqa: E402
from tools.human_optim.qmath import normalize_q  # noqa: E402
from tools.smplx_robot import load_smplx_mid  # noqa: E402

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "human_optimization.toml"
_DEFAULT_SMPLX_MODEL = (
    _REPO_ROOT.parent / "better_human" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
)

# Carve-hole probe constants (fixed — this is a fixed diagnostic, not a tunable stage).
_SLOW_MAX_MS = 0.2       # m/s below which a wrist counts as "slow" (a plausible contact)
_BALL_RADIUS_M = 0.10    # count conf-gated scene points inside this ball of the wrist
_CARVE_MIN_PTS = 20      # < this many nearby scene points ⇒ the hold looks carved away

_FINGER_TOKENS = ("index", "middle", "pinky", "ring", "thumb")


def _joint_class(name: str) -> str:
    """Coarse contact class for a SMPLXMid joint name: ``hands`` / ``feet`` / ``other``."""
    if "wrist" in name or any(tok in name for tok in _FINGER_TOKENS):
        return "hands"
    if "ankle" in name or "foot" in name:
        return "feet"
    return "other"


def _load_contacts_cfg(path: Path) -> dict:
    """The ``[contacts]`` knobs this diagnostic needs (thresholds + scene conf gate)."""
    cfg = tomllib.load(path.open("rb")) if path.exists() else {}
    c = cfg.get("contacts", {})
    return dict(
        close_thresh_m=float(c.get("close_thresh_m", 0.02)),
        exit_thresh_m=float(c.get("exit_thresh_m", 0.035)),
        conf_drop_q=float(c.get("conf_drop_q", 0.1)),
    )


@torch.no_grad()
def _pose_joints(mid, betas: torch.Tensor, q: torch.Tensor, scale: float, *, chunk: int = 64):
    """World-frame joint positions ``(N, J, 3)`` — the ``estimate_contacts._pose`` recipe.

    Re-applies the per-person global ``scale`` about the pelvis (joint 0), exactly
    as the contact stage / viewers do, so the wrist positions match the ones the
    contact SDF saw. Vertices are not needed here, so only joints are returned.
    """
    num_frames = q.shape[0]
    out_joints = []
    for start in range(0, num_frames, chunk):
        end = min(start + chunk, num_frames)
        out = mid.forward(betas[None].expand(end - start, -1),
                          normalize_q(q[start:end], mid.num_joints))
        jw = out.joints_world.tensor()[..., :3]
        root = jw[:, :1, :]
        out_joints.append(root + scale * (jw - root))
    return torch.cat(out_joints, 0)


# ─────────────────────────── panel 1: histograms ─────────────────────────

def _smooth(counts: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smooth of histogram counts (numpy only — no scipy dep)."""
    if window <= 1 or counts.size < window:
        return counts.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(counts.astype(np.float64), kernel, mode="same")


def _bimodality(values_cm: np.ndarray, thresholds_cm: tuple[float, float], nbins: int = 60) -> dict:
    """Histogram ``values_cm`` and test for a touching / free two-lobe split.

    Returns the histogram (``counts``, ``edges``, ``centers``, ``smoothed``) plus
    ``bimodal`` (two prominent peaks separated by a real valley) and, if so,
    ``valley_cm`` (the smoothed-histogram minimum between the two tallest peaks)
    and ``modes_cm`` (their centres). A single-lobe class returns ``bimodal=False``.
    """
    close_cm = thresholds_cm[0]
    p1, p99 = np.percentile(values_cm, [1.0, 99.0])
    # Bound the low edge at -10 cm: the raw (unguarded) joint_sdf_cm channel carries
    # spurious deep-occluder penetrations far below the stage's own max_pen_m guard,
    # which would otherwise stretch the range and hide the meaningful near-0 lobes.
    lo = max(-10.0, min(-2.0, float(p1)))
    hi = min(60.0, max(float(p99), close_cm * 3.0))
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(values_cm, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth(counts)

    # local maxima above a prominence floor: a real second lobe is ≥10% as tall as the
    # dominant one (kills tiny diffuse-tail bumps being called a free lobe).
    floor = 0.10 * smoothed.max() if smoothed.max() > 0 else 0.0
    peaks = [i for i in range(1, len(smoothed) - 1)
             if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
             and smoothed[i] > floor]
    result = dict(counts=counts, edges=edges, centers=centers, smoothed=smoothed,
                  bimodal=False, valley_cm=None, modes_cm=None)
    if len(peaks) < 2:
        return result

    peaks.sort(key=lambda i: smoothed[i], reverse=True)
    a, b = sorted(peaks[:2])                              # the two tallest, ordered by position
    if abs(centers[b] - centers[a]) < 3.0:               # modes must be genuinely separated
        return result
    valley = a + int(np.argmin(smoothed[a:b + 1]))
    peak_min = min(smoothed[a], smoothed[b])
    if smoothed[valley] < 0.6 * peak_min:                # a real dip, not a shoulder
        result.update(bimodal=True, valley_cm=float(centers[valley]),
                      modes_cm=(float(centers[a]), float(centers[b])))
    return result


def _plot_person(oid, class_stats, thresholds_cm, out_png):
    """Stacked hands / feet / other distance histograms for one person."""
    close_cm, exit_cm = thresholds_cm
    classes = ["hands", "feet", "other"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    for ax, cls in zip(axes, classes):
        st = class_stats[cls]
        n = 0 if st is None else int(st["n"])
        ax.set_ylabel(f"{cls}\ncount")
        if st is None or n == 0:
            ax.text(0.5, 0.5, f"{cls}: no finite joint_sdf_cm samples",
                    transform=ax.transAxes, ha="center", va="center", color="0.5")
            continue
        bi = st["bi"]
        ax.stairs(bi["counts"], bi["edges"], fill=True, alpha=0.4, color="#1f77b4")
        ax.plot(bi["centers"], bi["smoothed"], color="#08306b", lw=1.3, label="smoothed")
        ax.axvline(close_cm, color="r", ls="--", lw=1.0, label=f"close={close_cm:g}cm")
        ax.axvline(exit_cm, color="orange", ls=":", lw=1.0, label=f"exit={exit_cm:g}cm")
        tag = "BIMODAL" if bi["bimodal"] else "single-lobe"
        if bi["bimodal"]:
            ax.axvline(bi["valley_cm"], color="green", ls="-", lw=1.4,
                       label=f"valley={bi['valley_cm']:.1f}cm")
        ax.set_title(f"{cls}  (n={n}, median={st['median']:.1f}cm)  —  {tag}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("per-joint min perpendicular signed distance (cm)")
    fig.suptitle(f"person obj_id={oid} — contact distance distributions by joint class", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _class_histograms(joint_sdf_cm, valid, class_of, thresholds_cm):
    """Per-class ``joint_sdf_cm`` pool (valid frames, NaN dropped) → stats + bimodality.

    ``joint_sdf_cm`` is ``(N, J)`` for one person. Returns ``{class: stats|None}``
    and, separately, which classes were entirely NaN (a schema surprise to report).
    """
    stats: dict[str, dict | None] = {}
    all_nan: list[str] = []
    valid_sdf = joint_sdf_cm[valid]                       # (Nv, J)
    for cls in ("hands", "feet", "other"):
        cols = [j for j, c in enumerate(class_of) if c == cls]
        pooled = valid_sdf[:, cols].reshape(-1)
        finite = pooled[np.isfinite(pooled)]
        if finite.size == 0:
            stats[cls] = None
            if valid_sdf[:, cols].size > 0:
                all_nan.append(cls)
            continue
        stats[cls] = dict(n=finite.size, median=float(np.median(finite)),
                          bi=_bimodality(finite, thresholds_cm))
    return stats, all_nan


# ───────────────────────── probe 2: carve-hole cost ──────────────────────

def _carve_probe(joints_world, joint_contact, valid, wrist_idx, wrist_names, fps,
                 scene_pts):
    """Per-wrist carve-hole rows for one person.

    For each wrist: slow (central-diff world speed < ``_SLOW_MAX_MS``) & not-in-contact
    & valid frames; count conf-gated scene points within ``_BALL_RADIUS_M`` of the
    wrist; a frame is "carved" when that count < ``_CARVE_MIN_PTS``. Returns a list of
    ``(name, n_frames, n_carved, median_pts)`` rows.
    """
    rows = []
    num_frames = joints_world.shape[0]
    for w, name in zip(wrist_idx, wrist_names):
        pos = joints_world[:, w, :]                                  # (N, 3) world
        speed = torch.full((num_frames,), float("nan"), device=pos.device)
        if num_frames >= 3:
            speed[1:-1] = (pos[2:] - pos[:-2]).norm(dim=-1) * (fps / 2.0)  # central diff → m/s
        speed_np = speed.cpu().numpy()
        slow = np.isfinite(speed_np) & (speed_np < _SLOW_MAX_MS)
        qualify = slow & (~joint_contact[:, w]) & valid              # slow, no-contact, valid
        idx = np.nonzero(qualify)[0]
        if idx.size == 0:
            rows.append((name, 0, 0, float("nan")))
            continue
        centers = pos[torch.as_tensor(idx, device=pos.device)]       # (Q, 3)
        d = torch.cdist(centers, scene_pts)                          # (Q, M)
        near = (d < _BALL_RADIUS_M).sum(dim=1).cpu().numpy()         # (Q,)
        n_carved = int((near < _CARVE_MIN_PTS).sum())
        rows.append((name, int(idx.size), n_carved, float(np.median(near))))
    return rows


# ─────────────────────────────── main ────────────────────────────────────

def run(predictions_dir, config_path, device):
    pred = Path(predictions_dir)
    cfg = _load_contacts_cfg(Path(config_path))
    thresholds_cm = (cfg["close_thresh_m"] * 100.0, cfg["exit_thresh_m"] * 100.0)

    kin_path = paths.body_path(pred, "kinematic_1")
    con_path = paths.contacts_path(pred, "1")
    scene_path = pred / "geometry" / "scene.npz"
    for pth in (kin_path, con_path, scene_path):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run the contact chain first")

    kin = dict(np.load(kin_path, allow_pickle=True))
    con = dict(np.load(con_path, allow_pickle=True))
    scn = dict(np.load(scene_path, allow_pickle=True))

    num_people, num_frames = int(kin["num_people"]), int(kin["num_frames"])
    fps = float(kin["fps"])
    object_ids = [int(x) for x in kin["object_ids"]]
    joint_names = [str(x) for x in kin["joint_names"]]
    class_of = [_joint_class(n) for n in joint_names]
    wrist_idx = [j for j, n in enumerate(joint_names) if n.endswith("wrist")]
    wrist_names = [joint_names[j] for j in wrist_idx]
    valid_all = np.asarray(kin["valid_mask"], dtype=bool)             # (P, N)

    # Scene cloud with the SAME quantile confidence gate estimate_contacts applies at load.
    scene_pts = torch.as_tensor(scn["points"], dtype=torch.float32, device=device)
    scene_conf = torch.as_tensor(scn["confidence"], dtype=torch.float32, device=device)
    q_drop = cfg["conf_drop_q"]
    if q_drop > 0.0 and scene_conf.numel():
        keep = scene_conf >= torch.quantile(scene_conf, q_drop)
        scene_pts = scene_pts[keep]
    print(f"[diagnose_contacts] {pred.name}: {num_people} person(s), {num_frames} frames, "
          f"scene {scene_pts.shape[0]} pts after conf_drop_q={q_drop:g} gate; "
          f"close={thresholds_cm[0]:g}cm exit={thresholds_cm[1]:g}cm")

    mid = load_smplx_mid(_DEFAULT_SMPLX_MODEL, device)
    diag_dir = paths.diagnostics_dir(pred)

    carve_rows_all = []
    for p, oid in enumerate(object_ids):
        valid = valid_all[p]
        # ---- panel 1: distance histograms per joint class ----
        stats, all_nan = _class_histograms(
            np.asarray(con["joint_sdf_cm"][p], dtype=np.float32), valid, class_of, thresholds_cm)
        out_png = diag_dir / f"contacts_hist_person{oid:02d}.png"
        _plot_person(oid, stats, thresholds_cm, out_png)
        print(f"[diagnose_contacts] wrote {out_png}")
        for cls in ("hands", "feet", "other"):
            st = stats[cls]
            if st is None:
                print(f"    {cls:5s}: NO finite joint_sdf_cm samples (schema surprise — reported)")
                continue
            bi = st["bi"]
            if bi["bimodal"]:
                print(f"    {cls:5s}: n={st['n']:6d} median={st['median']:6.1f}cm  "
                      f"BIMODAL valley={bi['valley_cm']:.1f}cm modes={bi['modes_cm'][0]:.1f}/"
                      f"{bi['modes_cm'][1]:.1f}cm")
            else:
                print(f"    {cls:5s}: n={st['n']:6d} median={st['median']:6.1f}cm  "
                      f"single-lobe (NO separation — contact not trustworthy)")
        if all_nan:
            print(f"    [schema] joint_sdf_cm all-NaN for class(es): {', '.join(all_nan)}")

        # ---- probe 2: carve-hole cost ----
        betas = torch.as_tensor(kin["betas"][p], dtype=torch.float32, device=device)
        q = torch.as_tensor(kin["q"][p], dtype=torch.float32, device=device)
        scale = float(kin["scale"][p])
        joints_world = _pose_joints(mid, betas, q, scale)
        joint_contact = np.asarray(con["joint_contact"][p], dtype=bool)  # (N, J)
        rows = _carve_probe(joints_world, joint_contact, valid, wrist_idx, wrist_names,
                            fps, scene_pts)
        for name, n, n_carved, med in rows:
            carve_rows_all.append((oid, name, n, n_carved, med))

    # ---- per-clip carve-hole summary table ----
    print(f"\n[diagnose_contacts] carve-hole probe (wrist slow<{_SLOW_MAX_MS}m/s & not-in-contact; "
          f"< {_CARVE_MIN_PTS} scene pts within {_BALL_RADIUS_M * 100:g}cm ⇒ carved):")
    print(f"    {'oid':>3s}  {'wrist':<12s}  {'n_slowfree':>10s}  {'n_carved':>8s}  "
          f"{'frac':>5s}  {'med_pts':>7s}")
    tot_n = tot_carved = 0
    for oid, name, n, n_carved, med in carve_rows_all:
        frac = (n_carved / n) if n else float("nan")
        med_s = f"{med:.0f}" if np.isfinite(med) else "-"
        print(f"    {oid:>3d}  {name:<12s}  {n:>10d}  {n_carved:>8d}  {frac:>5.2f}  {med_s:>7s}")
        tot_n += n
        tot_carved += n_carved
    clip_frac = (tot_carved / tot_n) if tot_n else float("nan")
    print(f"    {'ALL':>3s}  {'TOTAL':<12s}  {tot_n:>10d}  {tot_carved:>8d}  {clip_frac:>5.2f}")
    return clip_frac


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    run(args.predictions, args.config, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
