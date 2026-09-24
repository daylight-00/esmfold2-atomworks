"""Declared identities for the non-polymer parts of a structure.

A residue name is a *label*, not an identity. ``LIG`` is a real CCD code, and so
are ``UNL`` and ``UNK``; a model that emitted a SMILES ligand under the label
``LIG`` and a depositor who crystallized CCD ``LIG`` produce the same three
characters and different molecules. Reconciling the first against the CCD
silently replaces the ligand: in one observed case that turned 19 correct ligand
atoms into 8 carbons while leaving a structure that still parsed and still
validated.

So this module exists to make ligand identity something the caller *declares*
and this code *verifies*, rather than something inferred from a string. The
verification is deliberately about composition rather than names, because atom
names are the thing that differs between sources.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = [
    "ChainDeclarationError",
    "CovalentBondResolutionError",
    "InferredChainKindError",
    "LigandIdentityError",
    "LigandSpec",
    "MixedChainError",
    "ModificationResolutionError",
    "UnsupportedChainError",
    "formula_of",
]

#: Residue names that carry no reliable chemical meaning, even though each is a
#: real CCD code. A structure that uses one of these for a ligand is *not*
#: making a statement this code may act on.
GENERIC_LIGAND_NAMES = frozenset({"LIG", "UNL", "UNK", "UNX"})


class LigandIdentityError(ValueError):
    """A ligand's declared identity disagrees with the atoms present."""


class UnsupportedChainError(ValueError):
    """A chain the adapter cannot express reached it, and was not opted out of.

    Raised rather than dropped, because the direct API returns no report: a
    caller folding a structure with one unsupported chain would otherwise get a
    confident prediction of a smaller system, with nothing saying a chain went
    missing. Water is the exception -- it is dropped under the explicit
    ``drop_water`` policy.
    """


class ChainDeclarationError(ValueError):
    """A per-chain declaration that would not reach exactly the chain it names.

    ``chain_kinds``, ``sequences``, ``msas`` and ``ligands`` are statements
    about particular chains. One that names no chain, names a chain of a kind it
    cannot apply to, contradicts another declaration, or is not true of the
    chain it names would otherwise be ignored or misapplied: the caller believes
    something was applied that the model never sees. That is an invalid request
    rather than an approximation, so unlike the degradations there is no
    opt-in.
    """


class MixedChainError(ValueError):
    """One chain label holds atoms of more than one kind of molecule.

    ESMFold2 takes one input per chain, so a chain is classified as a whole. A
    chain whose atoms carry two ``chain_type`` values -- a protein and its
    ligand under one author chain id, say -- would be classified by one of them
    and the rest folded as part of it, or not at all. The remedy is a chain
    label per molecule, not an opt-in, so there is none.
    """


class InferredChainKindError(ValueError):
    """A chain's kind would have to be guessed because it has no ``chain_type``.

    ``is_polymer`` cannot tell protein from DNA or RNA, nor water from a
    ligand. A DNA duplex arriving without ``chain_type`` is otherwise folded as
    a protein of unknown residues -- measured: sequence ``XXXXXXXX``.
    """


class ModificationResolutionError(ValueError):
    """A non-standard residue could not be tied to a position in the sequence.

    Proceeding would fold the parent residue in its place -- selenomethionine as
    methionine, say -- which is a different molecule.
    """


class CovalentBondResolutionError(ValueError):
    """A covalent bond in the source could not be placed in the model's indexing.

    Raised rather than skipped, because skipping folds a connected system as
    though it were disconnected -- a confident prediction of a different
    molecule, with nothing in the output to say so. A caller that genuinely
    wants the unconnected reading asks for it with
    ``allow_unresolved_covalent_bonds=True``.
    """


def formula_of(atoms: AtomArray) -> dict[str, int]:
    """Heavy-atom composition, as ``element -> count``.

    Hydrogens are excluded on purpose. Whether they are present at all depends
    on the source: ``atomworks.io.parse`` hydrogenates by default (and the added
    hydrogens can carry NaN coordinates), Rosetta rebuilds them on load, and
    ESMFold2 reports none. Comparing heavy atoms is the only comparison that
    means the same thing on all three sides.
    """
    elements = [str(e).upper() for e in atoms.element]
    return dict(Counter(e for e in elements if e not in ("H", "D")))


@dataclass(frozen=True)
class LigandSpec:
    """The declared identity of one non-polymer chain.

    Exactly one of *smiles* or *ccd* identifies the molecule. ``chain_id`` names
    the chain this applies to in the source structure.

    Args:
        chain_id: chain label in the source ``AtomArray``.
        smiles: SMILES string; ESMFold2 generates a conformer from it.
        ccd: one or more CCD codes, for a ligand that genuinely is a CCD entry.
        residue_name: label to write on the resulting atoms. Defaults to the
            first CCD code, or to ``LIG`` for a SMILES ligand -- matching what
            ESMFold2 itself emits.
        expected_formula: heavy-atom composition to verify against. Optional;
            when given, :meth:`verify_against` enforces it.
    """

    chain_id: str
    smiles: str | None = None
    ccd: tuple[str, ...] | None = None
    residue_name: str | None = None
    expected_formula: dict[str, int] | None = field(default=None)

    def __post_init__(self) -> None:
        if (self.smiles is None) == (self.ccd is None):
            raise ValueError(
                f"ligand {self.chain_id!r}: declare exactly one of smiles= or ccd= "
                f"(got smiles={self.smiles!r}, ccd={self.ccd!r})"
            )
        if self.ccd is not None and not self.ccd:
            raise ValueError(f"ligand {self.chain_id!r}: ccd= is empty")

    @property
    def label(self) -> str:
        """The residue name to write on these atoms."""
        if self.residue_name is not None:
            return self.residue_name
        return self.ccd[0] if self.ccd else "LIG"

    def verify_against(self, atoms: AtomArray) -> None:
        """Check *atoms* against the declaration, raising on disagreement.

        Only checked when *expected_formula* was supplied: a declaration without
        one is a statement of identity that this code has no independent way to
        confirm, and inventing a check that always passes would be worse than
        having none.
        """
        if self.expected_formula is None:
            return
        observed = formula_of(atoms)
        if observed != dict(self.expected_formula):
            raise LigandIdentityError(
                f"ligand {self.chain_id!r} declared {self.expected_formula} but the "
                f"atoms present are {observed}. A residue name is a label, not an "
                "identity -- check that the declaration names the right molecule."
            )

    def to_esm(self, chain_id: str | None = None) -> Any:
        """The ESMFold2 ``LigandInput`` for this declaration."""
        from esm.models.esmfold2.types import LigandInput

        return LigandInput(
            id=chain_id if chain_id is not None else self.chain_id,
            smiles=self.smiles,
            ccd=list(self.ccd) if self.ccd else None,
        )
