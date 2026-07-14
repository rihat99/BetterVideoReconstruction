from __future__ import annotations

import numpy as np

from scripts.corpus import _is_fresh, _sha256


def test_contact_precompute_skip_requires_matching_provenance(tmp_path):
    body = tmp_path / "kinematic_1.npz"
    geom = tmp_path / "geometry"
    geom.mkdir()
    scene, transform = geom / "scene.npz", geom / "transform.npz"
    config = tmp_path / "config.toml"
    body.write_bytes(b"body")
    scene.write_bytes(b"scene")
    transform.write_bytes(b"transform")
    config.write_text("[contact_confidence]\n")
    out = tmp_path / "contacts_1.npz"
    np.savez(
        out,
        confidence_schema=np.int32(4),
        confidence_method=np.str_("motion-only-joint-label-v1"),
        source_body=np.str_(body.name),
        source_body_sha256=np.str_(_sha256(body)),
        source_scene_sha256=np.str_(_sha256(scene)),
        source_transform_sha256=np.str_(_sha256(transform)),
    )
    assert _is_fresh(out, body, geom, config)

    np.savez(
        out,
        confidence_schema=np.int32(3),
        confidence_method=np.str_("continuous-joint-label-v1"),
        source_body=np.str_(body.name),
        source_body_sha256=np.str_(_sha256(body)),
        source_scene_sha256=np.str_(_sha256(scene)),
        source_transform_sha256=np.str_(_sha256(transform)),
    )
    assert not _is_fresh(out, body, geom, config)

    np.savez(
        out,
        confidence_schema=np.int32(4),
        confidence_method=np.str_("motion-only-joint-label-v1"),
        source_body=np.str_(body.name),
        source_body_sha256=np.str_(_sha256(body)),
        source_scene_sha256=np.str_(_sha256(scene)),
        source_transform_sha256=np.str_(_sha256(transform)),
    )

    body.write_bytes(b"new body")
    assert not _is_fresh(out, body, geom, config)
