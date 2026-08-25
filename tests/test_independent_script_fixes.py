from __future__ import annotations

import os

import numpy as np

from scripts.corpus import (
    _CONTACT_LABEL_METHOD, _CONTACT_LABEL_SCHEMA, _geometry_metric, _is_fresh, _sha256,
)


def test_contact_precompute_skip_requires_matching_provenance(tmp_path):
    body = tmp_path / "kinematic_1.npz"
    geom = tmp_path / "geometry"
    geom.mkdir()
    scene, transform, human = geom / "scene.npz", geom / "transform.npz", geom / "human.npz"
    config = tmp_path / "config.toml"
    body.write_bytes(b"body")
    scene.write_bytes(b"scene")
    transform.write_bytes(b"transform")
    human.write_bytes(b"human")
    config.write_text("[contact_gate]\n")
    out = tmp_path / "contacts_1.npz"

    def write(schema: int, method: str) -> None:
        np.savez(
            out,
            contact_label_schema=np.int32(schema),
            contact_label_method=np.str_(method),
            source_body=np.str_(body.name),
            source_body_sha256=np.str_(_sha256(body)),
            source_scene_sha256=np.str_(_sha256(scene)),
            source_transform_sha256=np.str_(_sha256(transform)),
        )

    write(_CONTACT_LABEL_SCHEMA, _CONTACT_LABEL_METHOD)
    assert _is_fresh(out, body, geom, config)

    # A per-joint-era artifact (different keys entirely) is never fresh.
    np.savez(out, confidence_schema=np.int32(8),
             confidence_method=np.str_("source-conditional-v1"))
    assert not _is_fresh(out, body, geom, config)

    write(1, "completed-static-support-v1")
    assert not _is_fresh(out, body, geom, config)

    write(_CONTACT_LABEL_SCHEMA, _CONTACT_LABEL_METHOD)
    body.write_bytes(b"new body")
    assert not _is_fresh(out, body, geom, config)

    body.write_bytes(b"body")
    write(_CONTACT_LABEL_SCHEMA, _CONTACT_LABEL_METHOD)
    human.write_bytes(b"new human")  # optional input, but when present it counts
    later = out.stat().st_mtime + 10
    os.utime(human, (later, later))  # same-second writes: force a strictly newer input
    assert not _is_fresh(out, body, geom, config)

    # A truncated npz (killed mid-write) is stale, not a crash.
    write(_CONTACT_LABEL_SCHEMA, _CONTACT_LABEL_METHOD)
    out.write_bytes(out.read_bytes()[:40])
    assert not _is_fresh(out, body, geom, config)


def test_geometry_metric_tolerates_a_truncated_npz(tmp_path):
    geom = tmp_path
    tr = geom / "transform.npz"
    np.savez(tr, metric=np.bool_(True))
    assert _geometry_metric(geom)

    tr.write_bytes(tr.read_bytes()[:40])   # killed mid-write → not done, no crash
    assert not _geometry_metric(geom)
