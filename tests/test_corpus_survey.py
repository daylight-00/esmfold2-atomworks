"""The corpus survey, pinned per AtomWorks test file.

Each file reproduces or is refused with a named error on a named chain; an
unclassified file fails, so a new AtomWorks revision shows up here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: path under ``tests/data`` -> ``None`` (reproduces), or
#: ``(error type, text in the message)`` (refused).
EXPECTED: dict[str, tuple[str, str] | None] = {
    "io/101m_arginine_nh1nh2_swapped.cif": None,
    "io/1a8o_modified.cif": None,
    "io/1qfe.pdb": None,
    "io/2hhb.cif.gz": None,
    "io/6lyz.bcif": None,
    "io/7ubd_from_af3.cif": None,
    "io/8cjg_from_af3.cif": None,
    "io/UniRef50_A0A0S8JQ92_AF2_predicted.pdb": None,
    "io/example_conditional_generation_output.cif": None,
    "io/test_cif_loading_4q8n.cif.gz": None,
    "ml/af2_distillation/cif/7c/0b/UniRef50_A0A1H9L980.cif": None,
    "ml/af2_distillation/cif/ad/a2/UniRef50_A0A1Q4X5U9.cif": None,
    "ml/af2_distillation/cif/5c/75/UniRef50_UPI000A006E95.cif": None,
    "io/9cox_with_unknown_ccd.cif": (
        "LigandIdentityError",
        "chain 'C' is labelled ['UNKNOWN_CCD']",
    ),
    "io/example_distillation_output.cif": (
        "LigandIdentityError",
        "chain 'B' is labelled ['UNL']",
    ),
    "io/example_ncaa.cif": ("LigandIdentityError", "chain 'B' is labelled ['C:0']"),
    "io/test_unl_ligand_with_bonds.cif": (
        "LigandIdentityError",
        "chain 'A' is labelled ['UNL']",
    ),
}

#: Files of AtomWorks' own test data that its parser rejects (before this package
#: sees them), with the text of its error.
PARSE_MAY_REJECT = {
    "io/1a8o_modified.cif": "matches neither standard nor alternative CCD names",
}

_SUFFIXES = (".cif", ".cif.gz", ".bcif", ".pdb")


def _corpus(data_dir: Path) -> dict[str, Path]:
    """Every structure file under AtomWorks' ``tests/data``, by relative path."""
    root = data_dir.parent
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name.endswith(_SUFFIXES)
    }


def test_every_corpus_file_is_classified(data_dir):
    unclassified = sorted(set(_corpus(data_dir)) - set(EXPECTED))
    assert not unclassified, (
        f"AtomWorks' test corpus holds {unclassified}, which this table does not "
        "classify: run `esmfold2-atomworks parity` on them and add each outcome"
    )


@pytest.mark.slow
def test_survey_outcomes_are_as_pinned(data_dir, ccd):
    from esmfold2_atomworks.parity.run import parity_for_structure

    wrong: list[str] = []
    for key, path in _corpus(data_dir).items():
        if key not in EXPECTED:
            continue
        outcome = parity_for_structure(path)
        expected = EXPECTED[key]
        if expected is None:
            rejected_by_parse = (
                key in PARSE_MAY_REJECT and PARSE_MAY_REJECT[key] in outcome.detail
            )
            if not outcome.ok and not rejected_by_parse:
                wrong.append(f"{key}: expected to reproduce, got {outcome.detail}")
        else:
            error, text = expected
            if (
                outcome.ok
                or not outcome.detail.startswith(f"{error}:")
                or text not in outcome.detail
            ):
                wrong.append(f"{key}: expected {error} ({text}), got {outcome.detail}")
    assert not wrong, "\n".join(wrong)
