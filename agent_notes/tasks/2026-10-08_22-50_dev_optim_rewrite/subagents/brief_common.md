# Common context — BVR dev task (read-only investigation phase)

Workspace: /home/rikhat.akizhanov/better. Repos:
- BetterVideoReconstruction (BVR): main checkout /home/rikhat.akizhanov/better/BetterVideoReconstruction (just cleaned, HEAD 4e2c152); NEW dev worktree /home/rikhat.akizhanov/better/dev/BetterVideoReconstruction (branch dev, identical to main right now). Its CLAUDE.md describes the pipeline; agent_notes/tasks/2026-10-08_17-46_repo_cleanup/REPORT.md describes yesterday's cleanup.
- BetterRobot (BR): main /home/rikhat.akizhanov/better/BetterRobot; dev worktree /home/rikhat.akizhanov/better/dev/BetterRobot (branch dev). BR dev has a rewritten optimization module ("optim") — see its agent_notes/tasks/2026-10-06_20-13_architecture_plan and 2026-10-06_23-45_p1_optim and the code itself.
- BetterHuman (BH): main /home/rikhat.akizhanov/better/BetterHuman; dev worktree /home/rikhat.akizhanov/better/dev/BetterHuman (branch dev).

Goal of the parent task (on BVR dev, may be breaking, results must stay close to main): (1) BVR stops owning an optimization engine (tools/optim.py, the optimizer parts of tools/human_optim) and uses BR dev's optim; BVR only defines problems, variables and residuals/losses. (2) Update BVR to BR dev + BH dev APIs. (3) All dataset/benchmark-specific code (peter rigs, OpenCap, Parkour, BEDLAM2, AMASS, ClimbingVideos corpus, their configs, importers, scorers, exporters) leaves the generic core into per-dataset folders. (4) Remove everything unnecessary. "Super clean and usable".

Rules for this phase: read-only. Do not modify any repo. Verify claims by reading code (cite file:line). Prefer the actual code over notes (notes are prior agents' output). Write nothing to the repos; return your report as text in your final message (the harness may refuse report files) AND try to write it to the path in your brief. Open questions at the end.
