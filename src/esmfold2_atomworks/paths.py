"""Resolved filesystem locations for source trees, weights, and run outputs.

Everything is overridable by environment variable so the same code runs from the
repo, from a git worktree, and on a cluster where the trees are staged
elsewhere. ``reproducibility/env.sh`` sets the same variables for the shell.

``DESIGN_ROOT`` is *discovered* rather than fixed at ``REPO_ROOT.parent``: a git
worktree can live several levels below the repository, where a fixed parent
names the directory holding the worktree and every source tree would silently
vanish. Searching upward for the directory that actually holds the trees is
correct from both, and survives the next reorganization of the tree.
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
}

#: The trees the core needs: ESMFold2 itself, and the AtomWorks structures it
#: is fed from. Foundry backs the optional training integration (``training/``
#: and ``configs/``) and nothing else, so a workspace without it is complete
#: for everything the adapter, the model wrapper and the engine do.
REQUIRED_TREES = ("esm", "atomworks")
OPTIONAL_TREES = ("foundry",)

#: Directories whose presence identifies a candidate ``DESIGN_ROOT``.
_MARKERS = REQUIRED_TREES

REPO_ROOT = Path(__file__).resolve().parents[2]


def _discover_design_root(start: Path) -> Path:
    """The nearest ancestor of *start* holding every required source tree.

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
    "experimental": "biohub/ESMFold2-Experimental",
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
    experimental: Path = MODELS / "ESMFold2-Experimental"

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


#: The separately stored ESMC backbones a checkpoint may name, by Hub id, and
#: the directory under ``EF_MODELS`` that mirrors each. A mapping rather than a
#: rule, because a directory name is not an identity: ``otherorg/ESMC-6B`` must
#: never be served by the mirror of ``biohub/ESMC-6B``.
ESMC_MIRRORS: dict[str, str] = {
    "biohub/ESMC-6B": "ESMC-6B",
}


def resolve_esmc(esmc_id: str) -> Path | str:
    """Where the ESMC backbone a checkpoint names is, preferring a local mirror.

    A checkpoint either bundles its backbone (``config.esmc_config``) or names a
    separately stored one in ``config.esmc_id`` -- as a Hub id, or as whatever
    path was written into the ``config.json`` of the mirror it came from. Left
    to ``from_pretrained``, that string decides everything: an absolute path
    stops existing when the tree moves, and a Hub id downloads again although a
    mirror sits beside the checkpoint. So it is resolved here, by identity:

    1. a Hub id in :data:`ESMC_MIRRORS` -> its mirror under ``EF_MODELS``, when
       the mirror is present;
    2. any other Hub id (``org/name``) -> unchanged, for ``from_pretrained`` to
       fetch; no mirror stands in for a backbone it does not mirror;
    3. an existing directory -> itself;
    4. an absolute path that no longer exists -> the mirror its last component
       names, when that names exactly one known backbone. Such a path was
       written on another machine and records a directory, not an identity, so
       its name is all there is to go on.

    Raises:
        FileNotFoundError: for anything else, rather than handing
            ``from_pretrained`` a directory it will fail to find much later.
    """
    mirror = ESMC_MIRRORS.get(esmc_id)
    if mirror is not None and (MODELS / mirror).is_dir():
        return MODELS / mirror
    if _is_hub_id(esmc_id):
        return esmc_id
    as_path = Path(esmc_id).expanduser()
    if as_path.is_dir():
        return as_path
    if as_path.is_absolute():
        named = [name for name in ESMC_MIRRORS.values() if name == as_path.name]
        if len(named) == 1 and (MODELS / named[0]).is_dir():
            return MODELS / named[0]
    raise FileNotFoundError(
        f"the checkpoint names its ESMC backbone as {esmc_id!r}: not a Hub id, "
        f"not a directory, and not the name of a mirror present under {MODELS} "
        f"(mirrored: {sorted(ESMC_MIRRORS)}). Mirror the backbone there, or "
        "point EF_MODELS at the directory holding it."
    )


def _is_hub_id(name: str) -> bool:
    """``org/name``: exactly one slash, and not the start of a path."""
    return name.count("/") == 1 and not name.startswith(("/", ".", "~"))


def ccd_dir() -> Path | None:
    """The directory holding the workspace's ``ccd.pkl``, or ``None``.

    ESMFold2 builds ligand and modified-residue conformers from a pickled CCD
    published only in ``biohub/ESMFold2``, so the copy in that mirror serves
    every checkpoint. esm's ``load_ccd(cache_dir)`` reads
    ``<cache_dir>/ccd.pkl``; given nothing, it downloads the pickle from the
    Hub's latest revision into the Hugging Face cache -- a source that moves,
    and one that bypasses the mirror. Pass this instead. ``None`` means there
    is no mirror copy, and esm downloads as before. ``ESMCFOLD_CCD_PATH``, when
    set, still takes precedence inside esm.

    The dictionary is process-global and the first load wins, so this decides
    the source only when it reaches whichever call loads first -- and esm's own
    lazy lookups load with no location at all. ``reproducibility/env.sh``
    therefore also exports ``ESMCFOLD_CCD_PATH`` before Python starts:
    ``esm.models.esmfold2.conformers`` captures it when it is imported and
    prefers it over any location a caller passes, so whichever call loads
    first reads the mirror's pickle.
    """
    directory = ESMFOLD2_WEIGHTS.standard
    return directory if (directory / "ccd.pkl").is_file() else None


def pythonpath_entries() -> list[Path]:
    """The ``PYTHONPATH`` additions ``reproducibility/env.sh`` makes, in order."""
    return [*SOURCE_TREES.values(), REPO_ROOT / "src"]


def config_dir() -> Path:
    """The Hydra config directory, in whichever layout is present.

    The configs live at the repo root in a source checkout and inside the
    package once installed, because ``pkg://esmfold2_atomworks.configs`` has to
    resolve there. Callers
    that compose configs should ask here rather than assuming one of the two::

        with initialize_config_dir(config_dir=str(config_dir()), version_base="1.3"):
            cfg = compose(config_name="inference", overrides=[...])

    Raises:
        FileNotFoundError: when neither layout is present, which means the
            wheel was built without the configs -- the failure this function
            exists to make loud rather than mysterious.
    """
    packaged = Path(__file__).resolve().parent / "configs"
    if packaged.is_dir():
        return packaged
    source = REPO_ROOT / "configs"
    if source.is_dir():
        return source
    raise FileNotFoundError(
        f"no Hydra configs at {packaged} or {source}. In an installed package "
        "this means the wheel was built without the force-include that maps "
        "configs/ to esmfold2_atomworks/configs; see pyproject.toml."
    )


#: Trees whose revision ``UPSTREAM.lock`` records. The required trees must be
#: recorded; the optional one is recorded when the integration was verified
#: against it.
LOCKED_TREES = (*REQUIRED_TREES, *OPTIONAL_TREES)

UPSTREAM_LOCK = REPO_ROOT / "reproducibility" / "UPSTREAM.lock"


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
