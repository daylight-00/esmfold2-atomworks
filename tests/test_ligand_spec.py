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
