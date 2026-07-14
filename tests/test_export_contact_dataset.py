from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import export_contact_dataset as export


def test_hand_contacts_and_confidence_fold_to_wrists():
    contacts = np.zeros((1, 2, 52), bool)
    confidence = np.full((1, 2, 52), 0.2, np.float32)

    contacts[0, 0, 22] = True
    confidence[0, 0, 22] = 0.85
    contacts[0, 0, 21] = True
    confidence[0, 0, 21] = 0.7
    confidence[0, 1, export.LEFT_HAND_GROUP] = np.linspace(0.1, 0.4, 16)

    labels, conf = export.merge_contacts_52_to_22(contacts, confidence)

    assert labels.shape == conf.shape == (1, 2, 22)
    assert labels[0, 0, 20] and labels[0, 0, 21]
    assert conf[0, 0, 20] == pytest.approx(0.85)
    assert conf[0, 0, 21] == pytest.approx(0.7)
    assert not labels[0, 1, 20]
    assert conf[0, 1, 20] == pytest.approx(
        confidence[0, 1, export.LEFT_HAND_GROUP].mean()
    )


def test_hand_fold_rejects_bad_shape_and_confidence():
    with pytest.raises(ValueError, match="shape"):
        export.merge_contacts_52_to_22(
            np.zeros((1, 2, 51), bool), np.zeros((1, 2, 51), np.float32)
        )
    bad = np.zeros((1, 2, 52), np.float32)
    bad[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="within"):
        export.merge_contacts_52_to_22(np.zeros_like(bad, bool), bad)


def test_manual_mapping_is_by_name_and_only_14_joints_are_scored():
    names = list(reversed(export.ANNOTATION_JOINTS))
    ann_oids = [10, 20]
    data_oids = [20, 30, 10]
    values = np.full((2, len(names), 3), -1, np.int8)

    values[0, names.index("left_hand")] = [1, 0, -1]
    values[1, :, 0] = 0
    values[1, names.index("right_foot"), 1] = 1

    labels, annotated = export.map_annotation_to_22(
        values, names, ann_oids, data_oids, n_frames=3
    )

    # Dataset row 2 is annotation object 10, despite the different person order.
    np.testing.assert_array_equal(labels[2, :, 20], [True, False, False])
    np.testing.assert_array_equal(annotated[2, :, 20], [True, True, False])
    assert labels[0, 1, 11]
    assert annotated[0, 1, 11]
    assert not annotated[1].any()  # object 30 was never annotated

    observed = set(export.ANNOTATION_TO_SMPLX22.values())
    absent = [j for j in range(22) if j not in observed]
    assert not annotated[..., absent].any()


def test_ignored_flags_are_remapped_by_object_id():
    mask = export.ignored_mask_for_data(
        ann_oids=[7, 3], data_oids=[3, 9, 7], ignored=np.array([True, False])
    )
    np.testing.assert_array_equal(mask, [False, False, True])


def test_fill_test_labels_excludes_ignored_person(tmp_path: Path):
    sid = "abcd_scene"
    out_scene = tmp_path / "dataset" / "test" / sid
    ann_dir = tmp_path / "features" / "annotation" / "ab" / "cd" / sid
    out_scene.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    np.savez(
        out_scene / "inputs.npz",
        object_ids=np.array([3, 7], np.int32),
        frame_indices=np.arange(2, dtype=np.int32),
        joint_names_22=np.array([f"j{i}" for i in range(22)]),
        valid_mask=np.ones((2, 2), bool),
    )
    contacts = np.zeros((2, 14, 2), np.int8)
    np.savez(
        ann_dir / "annotation.npz",
        contacts=contacts,
        joint_names=np.array(export.ANNOTATION_JOINTS),
        object_ids=np.array([7, 3], np.int32),
        ignored=np.array([True, False]),
        num_frames=np.int64(2),
        annotation_version=np.int32(2),
    )

    summary = export.fill_test_labels(
        tmp_path / "dataset", tmp_path / "features", [sid]
    )
    result = np.load(out_scene / "contacts.npz")

    np.testing.assert_array_equal(result["ignored"], [False, True])
    assert result["annotated_22"][0].any()
    assert not result["annotated_22"][1].any()
    assert summary[0]["ignored_people"] == 1


def test_mask_copy_resumes_missing_files_without_overwriting(tmp_path: Path):
    source = tmp_path / "sam3" / "00"
    target = tmp_path / "masks" / "00"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    (source / "frame_000000.png").write_bytes(b"source-zero")
    (source / "frame_000001.png").write_bytes(b"source-one")
    (target / "000000.png").write_bytes(b"keep-me")

    export._copy_masks(
        tmp_path / "sam3", tmp_path / "masks", [0], 2, skip_existing=True
    )

    assert (target / "000000.png").read_bytes() == b"keep-me"
    assert (target / "000001.png").read_bytes() == b"source-one"


def test_mask_copy_fails_on_missing_or_stale_content(tmp_path: Path):
    source = tmp_path / "sam3" / "00"
    source.mkdir(parents=True)
    (source / "frame_000000.png").write_bytes(b"only-one")
    with pytest.raises(FileNotFoundError, match="frame_000001"):
        export._copy_masks(tmp_path / "sam3", tmp_path / "masks", [0], 2)

    stale = tmp_path / "other_masks" / "99"
    stale.mkdir(parents=True)
    with pytest.raises(ValueError, match="stale mask person"):
        export._copy_masks(tmp_path / "sam3", tmp_path / "other_masks", [0], 1)


def test_frame_extract_resumes_missing_jpeg(monkeypatch, tmp_path: Path):
    frames = [
        np.full((2, 3, 3), 10, np.uint8),
        np.full((2, 3, 3), 20, np.uint8),
    ]

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):
            return True

        def read(self):
            if self.index == len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(export.cv2, "VideoCapture", FakeCapture)
    out = tmp_path / "frames"
    out.mkdir()
    (out / "000000.jpg").write_bytes(b"keep-me")

    count = export._extract_frames(
        tmp_path / "clip.mp4", out, 2, 90, skip_existing=True
    )

    assert count == 2
    assert (out / "000000.jpg").read_bytes() == b"keep-me"
    assert (out / "000001.jpg").stat().st_size > 0


def test_exact_file_validation_rejects_stale_frame(tmp_path: Path):
    (tmp_path / "000000.jpg").write_bytes(b"ok")
    (tmp_path / "000001.jpg").write_bytes(b"stale")
    with pytest.raises(ValueError, match="extra"):
        export._validate_indexed_files(tmp_path, 1, ".jpg", "frames")


def test_contacts_frame_indices_must_be_sequential(tmp_path: Path):
    path = tmp_path / "contacts.npz"
    np.savez(path, frame_indices=np.array([0, 2], np.int32))
    with np.load(path) as contacts:
        with pytest.raises(ValueError, match="human_optim/contacts.frame_indices"):
            export._require_sequential_frames(
                "scene", "human_optim/contacts", contacts, n=2
            )


def test_skip_existing_preserves_completed_manual_contacts_byte_for_byte(tmp_path: Path):
    path = tmp_path / "contacts.npz"
    obj_ids = [3]
    n = 2
    joint_names = [f"joint_{j}" for j in range(22)]
    valid = np.array([[True, False]])
    labels = np.zeros((1, n, 22), bool)
    labels[0, 0, 20] = True
    annotated = np.zeros_like(labels)
    annotated[0, 0, 20] = True
    np.savez_compressed(
        path,
        joint_contact_22=labels,
        annotated_22=annotated,
        joint_names_22=np.asarray(joint_names),
        object_ids=np.asarray(obj_ids, np.int32),
        frame_indices=np.arange(n, dtype=np.int32),
        valid_mask=valid,
        ignored=np.array([False]),
        annotation_version=np.int32(2),
        pending=np.bool_(False),
    )
    before = path.read_bytes()

    ready = export._write_pending_test_contacts(
        path,
        sid="scene",
        obj_ids=obj_ids,
        n=n,
        joint_names=joint_names,
        valid=valid,
        skip_existing=True,
    )

    assert ready
    assert path.read_bytes() == before
    with np.load(path) as preserved:
        assert not bool(preserved["pending"])
        assert preserved["joint_contact_22"][0, 0, 20]
