"""Resolved filesystem locations for source trees, weights, and run outputs.

Everything is overridable by environment variable so the same code runs from the
repo, from a git worktree, and on a cluster where the trees are staged
elsewhere. ``env.sh`` sets the same variables for the shell.

``DESIGN_ROOT`` is *discovered* rather than fixed at ``REPO_ROOT.parent``: a git
worktree lives several levels below the repo, so a fixed parent would resolve to
``.claude/worktrees`` and every source tree would silently vanish. Searching
upward for the directory that actually holds the trees is correct from both, and
survives the next reorganization of the workspace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Top-level module each source tree provides -> path of the tree, relative to
#: ``DESIGN_ROOT``. This is the single declaration of what "the trees" means;
#: discovery, ``PYTHONPATH`` assembly and ``doctor`` all read it.
SOURCE_TREE_LAYOUT: dict[str, tuple[str, ...]] = {
    "esm": ("esm",),
    "atomworks": ("atomworks", "src"),
    "foundry": ("foundry", "src"),
    "mpnn": ("foundry", "models", "mpnn", "src"),
}

#: Directories whose presence identifies a candidate ``DESIGN_ROOT``.
_MARKERS = ("esm", "atomworks", "foundry")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _discover_design_root(start: Path) -> Path:
    """The nearest ancestor of *start* holding every source tree.

    Falls back to ``start.parent`` -- the layout when the trees are staged but
    incomplete. ``doctor`` then reports exactly which one is missing, which is a
    better failure than a confident path into an empty directory.
    """
    for candidate in (start, *start.parents):
        if all((candidate / marker).is_dir() for marker in _MARKERS):
            return candidate
    return start.parent


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


DESIGN_ROOT = _env_path("DESIGN_ROOT", _discover_design_root(REPO_ROOT))

MODELS = _env_path("EF_MODELS", DESIGN_ROOT / "biohub")
CHECKPOINTS = _env_path("EF_CHECKPOINTS", DESIGN_ROOT / "checkpoints")
RUNS = _env_path("EF_RUNS", REPO_ROOT / "runs")

#: Source trees that must be importable via ``PYTHONPATH`` rather than pip.
SOURCE_TREES: dict[str, Path] = {
    module: DESIGN_ROOT.joinpath(*parts) for module, parts in SOURCE_TREE_LAYOUT.items()
}


#: HuggingFace repository ids, used when no local mirror is present. Passing one
#: of these to ``from_pretrained`` downloads and caches it, so a fresh clone
#: works without any manual staging.
HUB_IDS: dict[str, str] = {
    "standard": "biohub/ESMFold2",
    "fast": "biohub/ESMFold2-Fast",
}


@dataclass(frozen=True)
class ESMFold2Weights:
    """Where the weights are, preferring a local mirror over the Hub.

    A local directory laid out by ``huggingface-cli download`` is used when it
    exists; otherwise the Hub id is returned and ``from_pretrained`` fetches it.
    Returning a path that does not exist would turn a missing download into a
    confusing "config.json not found" much later.
    """

    standard: Path = MODELS / "ESMFold2"
    fast: Path = MODELS / "ESMFold2-Fast"

    def resolve(self, name: str = "standard") -> Path | str:
        """The local directory if present, else the HuggingFace repo id."""
        try:
            local: Path = getattr(self, name)
        except AttributeError:
            raise ValueError(
                f"unknown ESMFold2 weight set {name!r}; known: {sorted(HUB_IDS)}"
            ) from None
        return local if local.is_dir() else HUB_IDS[name]


ESMFOLD2_WEIGHTS = ESMFold2Weights()


def pythonpath_entries() -> list[Path]:
    """The ``PYTHONPATH`` additions ``env.sh`` makes, in the same order."""
    return [*SOURCE_TREES.values(), REPO_ROOT / "src"]


#: Trees whose revision is recorded in ``UPSTREAM.lock``. ``mpnn`` lives inside
#: the foundry tree, so it is covered by foundry's entry.
LOCKED_TREES = ("esm", "atomworks", "foundry")

UPSTREAM_LOCK = REPO_ROOT / "UPSTREAM.lock"


def read_upstream_lock(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Parse ``UPSTREAM.lock`` into ``{tree: {key: value}}``.

    Hand-rolled rather than via ``configparser`` so the file can carry comment
    lines that explain themselves, which is most of its value.
    """
    path = path or UPSTREAM_LOCK
    if not path.exists():
        return {}
    locked: dict[str, dict[str, str]] = {}
    section = ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            locked[section] = {}
        elif "=" in line and section:
            key, _, value = line.partition("=")
            locked[section][key.strip()] = value.strip()
    return locked


def tree_revision(tree: str) -> str | None:
    """The checked-out commit of a source tree, or ``None`` if not a git repo."""
    import subprocess

    directory = SOURCE_TREES.get(tree)
    if directory is None:
        return None
    # esm's tree path is the repo root; the others point at src/, so walk up
    # until a .git turns up.
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            try:
                out = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            return out.stdout.strip() or None
    return None
