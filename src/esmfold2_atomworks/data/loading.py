"""Reading a structure file into the pair the adapter takes, and recording how.

``atomworks.io.parse`` is configured by a ``ParseConfig``, and its defaults
belong to the AtomWorks release. :func:`parse_structure` reads a file the way the
inference engine does; :func:`parse_provenance` states the AtomWorks version and
what differs from the default configuration, for a fold's record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atomworks.io.config import ParseConfig
    from biotite.structure import AtomArray

__all__ = [
    "atomworks_version",
    "parse_provenance",
    "parse_structure",
    "resolve_parse_config",
]


def atomworks_version() -> str:
    """The AtomWorks version in use, ``"unknown"`` when it cannot be read."""
    import atomworks

    return str(getattr(atomworks, "__version__", "unknown"))


def resolve_parse_config(parse_config: Any = None) -> ParseConfig:
    """The ``ParseConfig`` that *parse_config* stands for.

    ``None`` is AtomWorks' defaults; a string names one of its presets
    (``"rcsb"``, ...); a mapping gives fields, and a field that does not exist
    raises, so a misspelt option is not ignored; a ``ParseConfig`` is used as it is.
    """
    from atomworks.io.config import ParseConfig

    if parse_config is None:
        return ParseConfig()
    if isinstance(parse_config, ParseConfig):
        return parse_config
    if isinstance(parse_config, str):
        return ParseConfig.from_preset(parse_config)
    if isinstance(parse_config, Mapping):
        return ParseConfig(**parse_config)
    raise TypeError(
        "parse_config must be None, a preset name, a mapping of ParseConfig fields "
        f"or a ParseConfig, not {type(parse_config).__name__}"
    )


def parse_structure(
    source: str | Path, parse_config: Any = None
) -> tuple[AtomArray, dict]:
    """``(atom_array, chain_info)`` of the first model of *source*'s asymmetric unit.

    Args:
        source: a structure file.
        parse_config: what :func:`resolve_parse_config` takes.

    Raises:
        ValueError: ``parse`` returned one result per model, as it does for a
            multi-model file of variable topology. Picking one would fold a model
            nobody chose; parse the model you want and pass the ``AtomArray`` and
            ``chain_info`` instead.
    """
    from atomworks.io import parse

    result = parse(source, config=resolve_parse_config(parse_config))
    if isinstance(result, list):
        raise ValueError(  # noqa: TRY004 - a result shape, not a wrong argument type
            f"{source}: AtomWorks returned {len(result)} results, one per model "
            "(a multi-model file of variable topology). Parse the model you want "
            "and pass its AtomArray and chain_info."
        )
    return result["asym_unit"][0], result["chain_info"]


def parse_provenance(parse_config: Any = None) -> dict[str, Any]:
    """The AtomWorks version and the parse configuration, as JSON-safe entries.

    The configuration is what differs from AtomWorks' defaults, or ``"default"``
    when nothing does; the defaults themselves belong to the recorded version.
    """
    from atomworks.io.config import ParseConfig

    defaults = ParseConfig().to_dict()
    changed = {
        key: value
        for key, value in resolve_parse_config(parse_config).to_dict().items()
        if value != defaults[key]
    }
    return {
        "atomworks.version": atomworks_version(),
        "atomworks.parse_config": json.loads(
            json.dumps(changed or "default", default=str)
        ),
    }
