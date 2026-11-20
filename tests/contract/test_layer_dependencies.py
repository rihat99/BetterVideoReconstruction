"""The edge table: which package of this repository may import which.

``bvr`` is the core, ``bvr_datasets`` builds on it, ``scripts`` are entry points that nothing
imports. Inside the core the foundation (config, paths, files, manifest, camera) knows only itself,
stages meet through files and never import one another, and ``bvr.run`` starts stages by module
name without importing them. Each third-party model is imported by one stage module.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FOUNDATION = {"bvr.config", "bvr.paths", "bvr.files", "bvr.manifest", "bvr.camera"}
CORE = FOUNDATION | {"bvr.body", "bvr.geometry", "bvr.terms", "bvr.contacts"}

#: package -> the packages of this repository it may import, besides itself.
ALLOWED = {
    **{package: FOUNDATION for package in FOUNDATION},
    "bvr.body": {"bvr.files"},
    "bvr.geometry": {"bvr.camera"},
    "bvr.terms": {"bvr.camera", "bvr.body", "bvr.geometry"},
    "bvr.contacts": {"bvr.terms", "bvr.camera", "bvr.body", "bvr.geometry"},
    "bvr.stages": CORE | {"bvr.run"},  # bvr.run.main is the entry point of every stage module
    "bvr.run": {"bvr.config", "bvr.paths", "bvr.manifest"},
    "bvr.viewer": {"bvr.files", "bvr.paths", "bvr.body", "bvr.camera"},
    "bvr_datasets": CORE | {"bvr.stages", "bvr.run", "bvr.viewer"},
    "scripts": {"bvr.run", "bvr.viewer", "bvr.config", "bvr.paths", "bvr_datasets"},
}

#: package -> the sibling libraries it may import.
LIBRARIES = {
    "bvr.body": {"better_robot", "better_human"},
    "bvr.terms": {"better_robot", "better_human"},
    "bvr.contacts": {"better_robot"},
    "bvr.stages": {"better_robot", "better_human"},
    "bvr.viewer": {"better_robot", "better_human"},
    "bvr_datasets": {"better_robot", "better_human"},
}

#: The third-party models; each is imported by exactly one module, a stage.
MODELS = {"sam3", "sapiens", "vggt_omega", "sam_3d_body", "geocalib"}

#: Modules of the old core, not held to the table until the rebuild replaces them. An entry leaves
#: in the commit that ports or removes its module; the list is empty when the rebuild is done.
UNPORTED = {
    "bvr.old_config", "bvr.npz_io", "bvr.mask_io", "bvr.source", "bvr.determinism", "bvr.human_optim",
    "bvr.stages.fuse_geometry", "bvr.stages.estimate_scale", "bvr.stages.human_optimization",
    "bvr.stages.estimate_contacts", "bvr.stages.measure_contacts", "bvr.stages.build_support_scene",
    "bvr_datasets", "scripts.view_3d", "scripts.scene_quality",
}


def modules() -> dict[str, Path]:
    """Dotted name -> file, for every Python module of ``bvr``, ``bvr_datasets`` and ``scripts``."""
    found = {}
    for top in ("bvr", "bvr_datasets", "scripts"):
        for path in sorted((REPO_ROOT / top).rglob("*.py")):
            parts = path.relative_to(REPO_ROOT).with_suffix("").parts
            found[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return found


def package(module: str) -> str:
    """The row of the table a module belongs to."""
    parts = module.split(".")
    return ".".join(parts[:2]) if parts[0] == "bvr" and len(parts) > 1 else parts[0]


def imports(path: Path) -> set[str]:
    """Every module name a file imports, at any depth of the file."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.update({node.module, *(f"{node.module}.{alias.name}" for alias in node.names)})
    return names


def unported(module: str) -> bool:
    return any(module == entry or module.startswith(entry + ".") for entry in UNPORTED)


def test_every_import_is_an_edge_of_the_table() -> None:
    known = modules()
    violations = []
    for module, path in known.items():
        if unported(module):
            continue
        own = package(module)
        for name in imports(path):
            top = name.split(".")[0]
            if top in ("bvr", "bvr_datasets", "scripts"):
                target = package(name)
                if target != "bvr" and target != own and target not in ALLOWED[own]:
                    violations.append(f"{module} imports {name}")
                if own == target == "bvr.stages" and name in known and name not in (module, "bvr.stages"):
                    violations.append(f"{module} imports the stage {name}")
            elif top in ("better_robot", "better_human") and top not in LIBRARIES.get(own, ()):
                violations.append(f"{module} imports {top}")
    assert not violations, "\n".join(sorted(violations))


def test_each_model_is_imported_by_one_stage() -> None:
    importers: dict[str, set[str]] = {model: set() for model in MODELS}
    for module, path in modules().items():
        if not unported(module):
            for name in imports(path):
                if name.split(".")[0] in MODELS:
                    importers[name.split(".")[0]].add(module)
    wrong = {model: sorted(users) for model, users in importers.items()
             if len(users) > 1 or any(package(user) != "bvr.stages" for user in users)}
    assert not wrong, wrong


def test_unported_modules_still_exist() -> None:
    known = modules()
    gone = sorted(entry for entry in UNPORTED if not any(m == entry or m.startswith(entry + ".") for m in known))
    assert not gone, f"remove from UNPORTED: {gone}"
