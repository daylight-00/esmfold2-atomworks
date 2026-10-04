"""Reading a structure file into the pair the adapter takes, and recording how.

``atomworks.io.parse`` is configurable, and its defaults differ between
AtomWorks releases, so the same path can give different inputs under different
versions. :func:`parse_structure` reads a file the way the inference engine does;
:func:`parse_provenance` states the AtomWorks version and the parse
configuration that produced the input, for a fold's record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = ["atomworks_version", "parse_provenance", "parse_structure"]


def atomworks_version() -> str:
    """The AtomWorks version in use, ``"unknown"`` when it cannot be read."""
    import atomworks

    return str(getattr(atomworks, "__version__", "unknown"))


def parse_structure(
    source: str | Path, parse_config: Any = None
) -> tuple[AtomArray, dict]:
    """``(atom_array, chain_info)`` of the first model of *source*'s asymmetric unit.

    Args:
        source: a structure file.
        parse_config: ``None`` for ``parse``'s own defaults; a mapping, passed as
            keyword options (the form AtomWorks 2.x takes, and 3.x still
            accepts); or any other object, passed as ``config=`` (a
            ``ParseConfig`` in AtomWorks 3.x).

    Raises:
        ValueError: ``parse`` returned one result per model, as AtomWorks 3.x
            does for a multi-model file of variable topology. Picking one would
            fold a model nobody chose; parse the model you want and pass the
            ``AtomArray`` and ``chain_info`` instead.
    """
    from atomworks.io import parse

    if parse_config is None:
        result = parse(source)
    elif isinstance(parse_config, Mapping):
        result = parse(source, **dict(parse_config))
    else:
        result = parse(source, config=parse_config)
    if isinstance(result, list):
        raise ValueError(  # noqa: TRY004 - a result shape, not a wrong argument type
            f"{source}: AtomWorks returned {len(result)} results, one per model "
            "(a multi-model file of variable topology). Parse the model you want "
            "and pass its AtomArray and chain_info."
        )
    return result["asym_unit"][0], result["chain_info"]


def parse_provenance(parse_config: Any = None) -> dict[str, Any]:
    """The AtomWorks version and the parse configuration, as JSON-safe entries."""
    if parse_config is None:
        described: Any = "default"
    elif isinstance(parse_config, Mapping):
        described = dict(parse_config)
    elif hasattr(parse_config, "to_dict"):
        described = parse_config.to_dict()
    else:
        described = repr(parse_config)
    return {
        "atomworks.version": atomworks_version(),
        "atomworks.parse_config": json.loads(json.dumps(described, default=str)),
    }
