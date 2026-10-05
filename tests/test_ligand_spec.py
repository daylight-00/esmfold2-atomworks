"""``LigandSpec`` states what a ligand is, so a malformed statement must not pass.

Needs nothing but the standard library, so these run in the offline job.
"""

from __future__ import annotations

import pytest

from esmfold2_atomworks.data.spec import LigandSpec

pytestmark = pytest.mark.offline


def test_a_bare_ccd_string_is_one_component():
    spec = LigandSpec(chain_id="B", ccd="ATP")
    assert spec.ccd == ("ATP",)
    assert spec.label == "ATP"


def test_a_ccd_list_becomes_a_tuple():
    assert LigandSpec(chain_id="B", ccd=["NAG", "NAG"]).ccd == ("NAG", "NAG")


def test_exactly_one_of_smiles_and_ccd_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        LigandSpec(chain_id="B")
    with pytest.raises(ValueError, match="exactly one"):
        LigandSpec(chain_id="B", smiles="CCO", ccd="ETH")


def test_an_empty_declaration_is_refused():
    with pytest.raises(ValueError, match="empty"):
        LigandSpec(chain_id="B", ccd=())
    with pytest.raises(ValueError, match="empty"):
        LigandSpec(chain_id="B", smiles="  ")


def test_a_smiles_ligand_without_a_name_gets_the_name_atomworks_gives_one():
    from esmfold2_atomworks.data.spec import ligand_labels

    assert ligand_labels({"B": LigandSpec(chain_id="B", smiles="CCO")}) == {"B": "L:0"}


def test_distinct_smiles_count_up_and_one_molecule_keeps_one_name():
    from esmfold2_atomworks.data.spec import ligand_labels

    specs = {
        "B": LigandSpec(chain_id="B", smiles="CCO"),
        "C": LigandSpec(chain_id="C", smiles="CCN"),
        "D": LigandSpec(chain_id="D", smiles="CCO"),
    }
    assert ligand_labels(specs) == {"B": "L:0", "C": "L:1", "D": "L:0"}


def test_a_declared_name_or_ccd_code_is_kept_and_takes_no_number():
    from esmfold2_atomworks.data.spec import ligand_labels

    specs = {
        "B": LigandSpec(chain_id="B", smiles="CCO", residue_name="ETO"),
        "C": LigandSpec(chain_id="C", ccd=("ZN",)),
        "D": LigandSpec(chain_id="D", ccd=("NAG", "BMA")),
        "E": LigandSpec(chain_id="E", smiles="CCN"),
    }
    assert ligand_labels(specs) == {"B": "ETO", "C": "ZN", "D": None, "E": "L:0"}
