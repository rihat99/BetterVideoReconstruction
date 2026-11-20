"""The datasets and benchmarks BVR runs on, one package each — nothing in ``bvr`` imports them.

Each ``bvr_datasets/<name>/`` holds that dataset's importer, sensor/plate readers, scorers,
exporters, configs, viewer reader and tests, plus a README saying what the data is, where it
lives and how to import, run and score it. ``bvr`` is the generic core; the dataset packages
build on it.
"""
