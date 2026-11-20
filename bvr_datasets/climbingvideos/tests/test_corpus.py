"""The view that lets the pipeline read the stage-first corpus as a clip-first run root."""
from __future__ import annotations

import sqlite3

from bvr import paths
from bvr_datasets.climbingvideos import corpus, scene_db


def test_the_view_is_a_run_root_of_the_corpus(tmp_path):
    data = scene_db.ScenePaths(tmp_path / "corpus")
    scene = "-4dCpR9i6tQ_0025"                              # an id that starts with a dash, as many do
    video = data.scenes_root / "-4" / "dC" / f"{scene}.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    contacts = data.feature_dir("human_optim", scene) / "contacts_1.npz"
    contacts.parent.mkdir(parents=True)
    contacts.write_bytes(b"labels")
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT ? AS scene_id, ? AS rel_path", (scene, f"-4/dC/{scene}.mp4")).fetchone()

    root = tmp_path / "view"
    corpus.view(data, [row], root)
    corpus.view(data, [row], root, create={"geometry"})    # a second build adds, and keeps what is there

    clip, = paths.clips(root, tmp_path / "run", upstream=root)
    assert clip.name == scene and clip.video.read_bytes() == b"video"
    assert clip.find(paths.CONTACTS_1).read_bytes() == b"labels"   # read through the link, from the corpus
    assert (root / scene / "geometry").resolve() == data.feature_dir("geometry", scene)
    assert not (root / scene / "sam3").exists()                     # a stage the scene lacks is not linked
