"""The datasets and benchmarks BVR runs on, one package each — nothing in ``bvr`` imports them.

Each ``datasets/<name>/`` holds that dataset's importer, sensor/plate readers, scorers,
exporters, configs, viewer reader and tests, plus a README saying what the data is, where it
lives and how to import, run and score it. ``bvr`` is the generic core; the dataset packages
build on it through its hooks (``sensor_forces.npz`` → ``measure_contacts``, the
``--variant`` farms of ``bvr/runner.py`` and ``scripts/pipeline.py --dataset``, the viewer's ``read`` /
``ClipData.extras``).
"""
