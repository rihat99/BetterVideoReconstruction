"""Pick one people-free frame per camera for every climb_wall_3 camera set-up.

The cameras were static within an inclination (checked with ORB + homography:
0-1 px between recordings), except ``cam_3`` at 16 degrees, which was bumped
twice (three poses, 40-47 px apart). So there are five camera sets: ``00``,
``16`` (cam_3 pose of a_16_01 / b_16_01, the one the calibration video saw),
``16b`` (a_16_02, a_16_03, b_16_02), ``16c`` (b_16_03) and ``32``. The
climber ``b`` recordings start with nobody in shot, so frame 0 of those is
used. ``cam_3`` was mounted 90 degrees off: its frames are rotated to upright
(real "down" was the image's right edge, so a 90-degree clockwise turn).

Writes ``peter/climb_wall_3_gt/scene/<set>/frames/<cam>.png`` + ``frames.json``.

    uv run --no-sync python scripts/peter_gt/extract_frames.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
ORIG = REPO / "peter/climb_wall_3_orig"
OUT = REPO / "peter/climb_wall_3_gt/scene"
CAMS = ["cam_left", "cam_right", "cam_3"]

# set -> (recording per camera, calibration folder or None, recordings using this cam_3 pose)
SETS: dict[str, dict] = {
    "00": {"frames": {c: "b_00_02" for c in CAMS}, "calibration": "calibration_00",
           "recordings": ["a_00_01", "a_00_02", "b_00_01", "b_00_02"]},
    "16": {"frames": {"cam_left": "b_16_03", "cam_right": "b_16_03", "cam_3": "b_16_01"},
           "calibration": "calibration_16", "recordings": ["a_16_01", "b_16_01"]},
    "16b": {"frames": {"cam_left": "b_16_03", "cam_right": "b_16_03", "cam_3": "b_16_02"},
            "calibration": None, "recordings": ["a_16_02", "a_16_03", "b_16_02"]},
    "16c": {"frames": {c: "b_16_03" for c in CAMS}, "calibration": None, "recordings": ["b_16_03"]},
    "32": {"frames": {c: "b_32_02" for c in CAMS}, "calibration": "calibration_32",
           "recordings": ["a_32_01", "a_32_02", "b_32_01", "b_32_02"]},
}
FRAME = 0


def read_frame(video: Path, index: int) -> cv2.typing.MatLike:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video}")
    return frame


def upright(cam: str, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Rotate cam_3 to upright; the other cameras are already level."""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if cam == "cam_3" else frame


def main() -> None:
    for name, spec in SETS.items():
        out = OUT / name / "frames"
        out.mkdir(parents=True, exist_ok=True)
        for cam, rec in spec["frames"].items():
            cv2.imwrite(str(out / f"{cam}.png"), upright(cam, read_frame(ORIG / rec / f"{cam}.mkv", FRAME)))
        meta = {
            "frame_index": FRAME,
            "source_recording": spec["frames"],
            "cam_3_rotation": "ROTATE_90_CLOCKWISE applied to the raw frame (raw 1920x1080 -> upright 1080x1920)",
            "calibration": spec["calibration"],
            "recordings": spec["recordings"],
        }
        (out / "frames.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{name}: {spec['frames']} -> {out}")


if __name__ == "__main__":
    main()
