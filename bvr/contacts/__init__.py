"""The contact readout: which contact frames of a body touch the scene, and how far to trust the answer.

Contact lives on the body's contact frames, the named skin points of :mod:`bvr.body`. Two witnesses
answer, and the readout unions them. ``detector`` is the geometric one: a frame touches where it is
near the scene and still. ``witness`` is the physical one: at the fixed pose, which frames must
carry load for the motion to be possible. ``labels`` holds the rules that turn solved load into
labels and fuse the two, and ``confidence`` says how often a label agrees with a human annotator
and how far a body's solved forces can be trusted.

What is measured runs on the device, on tensors. The rules that turn measurements into labels are
scans along time and run in numpy, on one person: arrays are ``(T, F)``, frames by contact frames.
"""
