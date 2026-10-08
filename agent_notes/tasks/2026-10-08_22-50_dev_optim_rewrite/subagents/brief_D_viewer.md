Read brief_common.md first.
# Investigation D — BetterRobot dev viewer tools vs BVR's viewer
Report path: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/agent_notes/tasks/2026-10-08_22-50_dev_optim_rewrite/subagents/inv_D_viewer.md
1. BR dev viewer: find the visualization/viewer module(s) in /home/rikhat.akizhanov/better/dev/BetterRobot (grep viser / viewer / visualize / render / scene in src, docs, examples, tests, agent_notes). Describe the public API: what it draws (robot meshes, point clouds, cameras, frames, trajectories, forces/arrows, contact markers, ground plane, video pane?), time scrubbing / playback, layers and toggles, GUI widgets, dataset browsing, screenshots/video export, how a user plugs in custom geometry. Cite file:line and a minimal example from BR's own examples/tests.
2. BVR viewer today: /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction/scripts/view_3d.py + tools/viewer/{app,helpers,loading,scene}.py. List every feature (body sources dropdown, contact frames layer, forces, scene normals, ground plane, video pane, scene-DB pager, BEDLAM mode, multi-view cameras, plate forces, ...) with file:line and roughly how many lines each costs.
3. Mapping table: BVR viewer feature → BR dev viewer equivalent (exists / partial / missing). For missing ones: generic enough to add to BR, or a thin BVR layer? Estimate how much of tools/viewer would disappear if BVR built its viewer on BR's.
4. Which BVR viewer features are dataset-specific (plate forces, BEDLAM browsing, scene-DB pager, athletics) and belong in datasets/<name>/ after the split; which are generic (one out/<stem>/ tree).
5. Tests touching the viewer (tests/test_viewer_*.py): what they pin.
Under 150 lines, tables.
