# Common context — repo cleanup audit (read-only)

Repo: /home/rikhat.akizhanov/better/BetterVideoReconstruction (git, branch main). Python project run via `uv run` from the repo root.
Goal of the parent task: clean the repo without breaking anything. Main optimization code (tools/human_optim, tools/optim.py, scripts/stages/*) is NOT to be modified; everything else (configs/, scripts/, tools/ helpers, root files) is being audited for removal/consolidation.
Planned moves (not done yet): repo-local dirs `data/`, `data_2/`, `out/`, `out_2/` → /home/rikhat.akizhanov/better/data/examples/<same name>; `peter/` → /home/rikhat.akizhanov/better/data/peter. Dirs `benchmark_tables/`, `experiments/`, `scratch/` will be removed.
Note /home/rikhat.akizhanov/better/data already exists and holds datasets (BEDLAM2, ClimbingVideos, AMASS, LabValidation_withVideos, Parkour-dataset, ...). Scripts may already reference it via absolute paths or `../data`.

Rules: read-only. Do not modify, move or delete anything. Do not run GPU jobs. Verify every claim by reading the file or running grep / git; cite file:line. Write your report to the path given in your brief, as markdown, concise, tables where useful. Put open questions at the end of the report.
