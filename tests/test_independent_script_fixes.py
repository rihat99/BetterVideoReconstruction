from __future__ import annotations

import numpy as np

from scripts.corpus import _CONFIDENCE_METHOD, _CONFIDENCE_SCHEMA, _is_fresh, _sha256


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
    config.write_text("[contact_confidence]\n")
    out = tmp_path / "contacts_1.npz"

    def write(schema: int, method: str) -> None:
        np.savez(
            out,
            confidence_schema=np.int32(schema),
            confidence_method=np.str_(method),
            source_body=np.str_(body.name),
            source_body_sha256=np.str_(_sha256(body)),
            source_scene_sha256=np.str_(_sha256(scene)),
            source_transform_sha256=np.str_(_sha256(transform)),
            source_human_sha256=np.str_(_sha256(human)),
        )

    write(_CONFIDENCE_SCHEMA, _CONFIDENCE_METHOD)
    assert _is_fresh(out, body, geom, config)

    write(3, "continuous-joint-label-v1")
    assert not _is_fresh(out, body, geom, config)

    write(_CONFIDENCE_SCHEMA, _CONFIDENCE_METHOD)
    body.write_bytes(b"new body")
    assert not _is_fresh(out, body, geom, config)

    body.write_bytes(b"body")
    write(_CONFIDENCE_SCHEMA, _CONFIDENCE_METHOD)
    human.write_bytes(b"new human")
    assert not _is_fresh(out, body, geom, config)
