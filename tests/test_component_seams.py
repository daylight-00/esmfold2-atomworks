"""``.esmc`` answers truthfully.

The wrapper itself needs the backbone: the LM prior, the provenance and a
separately published checkpoint all depend on whether one is attached. It can fail
by returning a plausible value -- ``None`` read as "no backbone attached" -- so it
is pinned here against the layouts the native module has.

The fakes need nothing installed and run in the offline job.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2


def _wrapper(net) -> AtomWorksESMFold2:
    """A wrapper around ``net`` without loading anything."""
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    model.net = net
    return model


# -- the ESMC backbone -------------------------------------------------------


@pytest.mark.offline
def test_esmc_is_read_under_the_native_name():
    backbone = object()
    assert _wrapper(SimpleNamespace(esmc=backbone)).esmc is backbone


@pytest.mark.offline
def test_esmc_falls_back_to_a_private_layout():
    backbone = object()
    assert _wrapper(SimpleNamespace(_esmc=backbone)).esmc is backbone


@pytest.mark.offline
def test_esmc_is_none_only_when_nothing_is_attached():
    # The native module keeps flags beside the backbone; neither is it.
    net = SimpleNamespace(esmc=None, _esmc_fp8=False, _offload_esmc=False)
    assert _wrapper(net).esmc is None
