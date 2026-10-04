"""Assembly copies that share a ``chain_id`` must not be folded as one chain.

A built assembly repeats a chain id under each ``transformation_id``; the key that
tells the copies apart is ``chain_iid``. Reading the structure by ``chain_id``
would hand ESMFold2 one chain for what is a homodimer.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("atomworks")
pytest.importorskip("biotite.structure")


def _homodimer_assembly():
    import biotite.structure as struc

    array = struc.AtomArray(4)
    array.coord = np.arange(12, dtype=np.float32).reshape(-1, 3)
    array.set_annotation("chain_id", np.array(["A"] * 4, dtype="U4"))
    array.set_annotation("res_id", np.array([1, 1, 1, 1]))
    array.set_annotation("ins_code", np.array([""] * 4, dtype="U1"))
    array.set_annotation("res_name", np.array(["GLY"] * 4, dtype="U5"))
    array.set_annotation("atom_name", np.array(["N", "CA", "N", "CA"], dtype="U6"))
    array.set_annotation("element", np.array(["N", "C", "N", "C"], dtype="U2"))
    array.set_annotation("is_polymer", np.ones(4, dtype=bool))
    array.set_annotation(
        "transformation_id", np.array(["1", "1", "2", "2"], dtype="U4")
    )
    array.set_annotation(
        "chain_iid", np.array(["A_1", "A_1", "A_2", "A_2"], dtype="U8")
    )
    return array


def test_copies_sharing_a_chain_id_are_refused():
    from esmfold2_atomworks.data.atomworks_to_esm import chain_records
    from esmfold2_atomworks.data.spec import MixedChainError

    with pytest.raises(MixedChainError, match="assembly copies"):
        chain_records(_homodimer_assembly())


def test_chain_iid_gives_each_copy_its_own_chain():
    from esmfold2_atomworks.data.atomworks_to_esm import chain_records

    records = chain_records(_homodimer_assembly(), chain_key="chain_iid")
    assert [record.chain_id for record in records] == ["A_1", "A_2"]
