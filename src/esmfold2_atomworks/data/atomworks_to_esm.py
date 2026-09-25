"""``AtomWorks AtomArray`` -> ``ESMFold2 StructurePredictionInput``.

This is the core of the package, and all the parity milestone needs: with
it, any AtomWorks-sourced example -- PDB, AFDB, a synthetic dimer, a PLINDER
protein-ligand pair -- can be folded by the *unmodified* ESMFold2, and the
result compared against what the native path produces.

**Why the adapter targets ``StructurePredictionInput`` and not the tensors.**
Both stacks featurize to an AF3-like layout, and the names overlap enough to
look interchangeable. They are not::

    AtomWorks                          ESMFold2
    restype            (N, 32) one-hot res_type           (N,)   index
    atom_to_token_map  (A,)             atom_to_token      (A,)
    ref_element        (A, 128) one-hot ref_element        (A,)   atomic number
    ref_atom_name_chars(A, 4, 64)       ref_atom_name_chars(A, 4) ord(c) - 32
    is_protein/rna/dna/ligand           mol_type           (N,)   enum
    msa_stack          (R, M, N, 34)    msa/has_deletion/deletion_value, split

Rebuilding ESMFold2's 29 feature tensors from AtomWorks' would mean reproducing
every one of those conventions exactly, and a subtle mismatch produces a model
that runs, reports plausible confidence, and is wrong. ``StructurePredictionInput``
is the declarative layer *above* both -- chains, sequences, ligand identities --
so converting there lets ESMFold2's own ``prepare_esmfold2_input`` build the
tensors it expects. Parity then becomes checkable rather than hoped for
(``esmfold2_atomworks.parity``).

**What is deliberately not inferred.** Chain classification comes from
AtomWorks' own ``chain_type`` annotation -- or, for an array without one, from
the caller's declaration, verified against the CCD -- and ligand identity from
an explicit declaration or a CCD code that is verified, never from a
residue-name guess. See :mod:`esmfold2_atomworks.data.spec`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from esmfold2_atomworks.data.spec import (
    GENERIC_LIGAND_NAMES,
    ChainDeclarationError,
    CovalentBondResolutionError,
    InferredChainKindError,
    LigandIdentityError,
    LigandSpec,
    MixedChainError,
    ModificationResolutionError,
    UnsupportedChainError,
)

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = [
    "DECLARABLE_CHAIN_KINDS",
    "DEGRADATIONS",
    "AdapterReport",
    "ChainRecord",
    "allow_kwargs",
    "atom_array_to_structure_prediction_input",
    "chain_records",
    "modifications_for_chain",
    "sequence_of_chain",
]


# --------------------------------------------------------------------------
# Strictness
# --------------------------------------------------------------------------
#: Every semantic degradation the adapter can detect, and the error it raises
#: by default. Each is permitted only by naming it: ``allow_<name>=True`` on the
#: adapter, ``allow=[<name>]`` in a pipeline or engine config, ``--allow <name>``
#: on the command line.
#:
#: The rule is that none of these happens silently. A degradation that is
#: merely *recorded* is invisible on the direct API, which returns no report,
#: so recording is never enough on its own. Keeping the complete list in one
#: table is what lets a test check that each one has an opt-in and defaults to
#: raising, rather than relying on remembering to add both.
#:
#: Not listed, deliberately: dropping water (an explicit ``drop_water`` policy,
#: not a degradation), and taking the sequence from modelled residues when no
#: ``chain_info`` is supplied. The latter can delete unresolved loops (D-003),
#: but without an entity record the adapter cannot establish that anything is
#: missing -- a numbering gap may be legitimate -- let alone what, and a policy
#: can only act on what it can establish.
DEGRADATIONS: dict[str, type[Exception]] = {
    "unsupported_chains": UnsupportedChainError,
    "unresolved_covalent_bonds": CovalentBondResolutionError,
    "inferred_chain_kind": InferredChainKindError,
    "unplaceable_modifications": ModificationResolutionError,
}


def allow_kwargs(names: object = ()) -> dict[str, bool]:
    """``["unsupported_chains"] -> {"allow_unsupported_chains": True}``.

    For the config- and CLI-driven paths, where a list of names is easier to
    write than a set of booleans. Unknown names raise rather than being
    ignored: a misspelt opt-in that silently does nothing would leave the
    caller believing they had accepted a degradation they had not -- or, worse,
    believing a strict run was permissive.
    """
    if names is None:
        return {}
    if isinstance(names, str):
        names = [names]
    unknown = sorted(set(names) - set(DEGRADATIONS))  # type: ignore[arg-type]
    if unknown:
        raise ValueError(
            f"unknown degradation(s) {unknown}; known: {sorted(DEGRADATIONS)}"
        )
    return {f"allow_{name}": True for name in names}  # type: ignore[union-attr]


def _remedy(name: str) -> str:
    """How to opt in, phrased for every path that can reach this error."""
    return (
        f"Opt in deliberately with allow_{name}=True (adapter / fold_atom_array "
        f"adapter_kwargs), allow=[{name!r}] (pipeline or engine config), or "
        f"--allow {name} (CLI)."
    )


def _declare_hint(chain_id: str) -> str:
    """The fix for a caller who knows what an unannotated chain is."""
    return (
        f"If you know what the chain is, declare it: chain_kinds={{{chain_id!r}: "
        f"<kind>}}, one of {sorted(DECLARABLE_CHAIN_KINDS)}."
    )


# --------------------------------------------------------------------------
# Chain classification
# --------------------------------------------------------------------------
# ESMFold2 has exactly four input classes. AtomWorks' ChainType is finer, so the
# mapping is declared once, here, from AtomWorks' own enum -- not re-derived
# from residue names, which is the guess this module exists to avoid.


def _chain_type_groups() -> tuple[
    frozenset[int], frozenset[int], frozenset[int], frozenset[int]
]:
    """``(protein, dna, rna, non_polymer)`` chain-type values.

    Read off ``atomworks.enums.ChainType`` so that a new chain type upstream
    shows up as "unclassified" -- and is reported -- rather than being silently
    folded into whichever group happened to be the default.
    """
    from atomworks.enums import ChainType

    protein = {
        ChainType.POLYPEPTIDE_L,
        ChainType.POLYPEPTIDE_D,
        ChainType.CYCLIC_PSEUDO_PEPTIDE,
    }
    dna = {ChainType.DNA}
    rna = {ChainType.RNA}
    non_polymer = {
        ChainType.NON_POLYMER,
        ChainType.BRANCHED,
        ChainType.MACROLIDE,
    }
    return (
        frozenset(int(c) for c in protein),
        frozenset(int(c) for c in dna),
        frozenset(int(c) for c in rna),
        frozenset(int(c) for c in non_polymer),
    )


#: Chain types ESMFold2 has no input for. Water is dropped by every structure
#: pipeline; it is listed explicitly so that dropping it is a decision with a
#: name, recorded in the report, rather than an accident.
def _droppable_chain_types() -> frozenset[int]:
    from atomworks.enums import ChainType

    return frozenset({int(ChainType.WATER)})


@dataclass(frozen=True)
class ChainRecord:
    """One chain of the source structure, classified and (if polymer) read out."""

    chain_id: str
    kind: str  # "protein" | "dna" | "rna" | "ligand" | "water" | "unsupported"
    n_atoms: int
    n_residues: int
    sequence: str | None = None
    residue_names: tuple[str, ...] = ()
    residue_ids: tuple[int, ...] = ()
    ins_codes: tuple[str, ...] = ()
    chain_type: int | None = None
    #: True when ``kind`` was inferred from ``is_polymer`` because the
    #: array carries no ``chain_type``. A nucleic-acid chain arriving that
    #: way is called protein, so the flag distinguishes an inference from
    #: a statement -- and the adapter will not fold on an inference unless
    #: it is accepted by name (``allow_inferred_chain_kind``).
    kind_is_inferred: bool = False

    @property
    def is_polymer(self) -> bool:
        return self.kind in ("protein", "dna", "rna")


@dataclass
class AdapterReport:
    """What the adapter did: the audit trail, not the safeguard.

    A conversion that quietly discards a chain, a bond or a modification is the
    failure mode this whole project is trying to make impossible: the fold
    still succeeds, the metrics still look reasonable, and the model was given
    a different system than the caller believes. Recording that here would not
    prevent it -- ``fold_atom_array`` returns no report -- so each such
    degradation raises instead (:data:`DEGRADATIONS`).

    The degradation fields are therefore filled only when the caller accepted
    one by name, and then say exactly what happened to the input returned.
    After a raise, the error message carries the detail. Every chain that does
    not reach ESMFold2 is named in ``dropped``.
    """

    chains: list[ChainRecord] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (chain_id, reason)
    sequence_source: dict[str, str] = field(default_factory=dict)
    unknown_residues: dict[str, list[str]] = field(default_factory=dict)
    modifications: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    #: Chains carrying non-standard residues whose position could not be tied
    #: to the folded sequence, where ``allow_unplaceable_modifications``
    #: accepted folding the parent residues instead -- a knowing
    #: approximation of the chemistry.
    unplaceable_modifications: list[str] = field(default_factory=list)
    #: Covalent bonds detected in the source structure, placed or not.
    covalent_bonds: list[Any] = field(default_factory=list)
    #: Bonds detected but not placeable in the model's indexing, with the
    #: reason, where ``allow_unresolved_covalent_bonds`` accepted folding
    #: without them. Never guessed -- a wrong index bonds the wrong pair of
    #: atoms, which upstream cannot detect.
    unresolved_covalent_bonds: list[str] = field(default_factory=list)
    #: Chains whose residues carry insertion codes that cannot be tied to
    #: the sequence, because ``chain_info`` does not record them. The fold
    #: input is unaffected; a covalent bond on such a chain cannot be placed
    #: (and so raises by default), and its labels are skipped rather than
    #: mis-assigned.
    unrepresentable_insertion_codes: list[str] = field(default_factory=list)
    #: Chains whose kind was inferred from ``is_polymer`` for want of a
    #: ``chain_type`` -- populated only when that was explicitly allowed.
    inferred_chain_kinds: list[str] = field(default_factory=list)

    def accepted_degradations(self) -> dict[str, list[str]]:
        """What each degradation accepted by name actually hit.

        Keyed like :data:`DEGRADATIONS` -- the same names a caller opts in
        with -- and empty when nothing was accepted. Water is not listed:
        dropping it is a policy, not a degradation.
        """
        return {name: hit for name, hit in self._degradations().items() if hit}

    def _degradations(self) -> dict[str, list[str]]:
        # One entry per DEGRADATIONS name, empty or not. tests/test_strictness.py
        # holds the two to the same keys, so a degradation cannot be added
        # without saying where its acceptance is recorded.
        return {
            "unsupported_chains": [
                chain for chain, reason in self.dropped if reason != "water"
            ],
            "unresolved_covalent_bonds": list(self.unresolved_covalent_bonds),
            "inferred_chain_kind": list(self.inferred_chain_kinds),
            "unplaceable_modifications": list(self.unplaceable_modifications),
        }

    def dropped_summary(self) -> str:
        if not self.dropped:
            return "nothing dropped"
        return "; ".join(f"{cid}: {why}" for cid, why in self.dropped)


# --------------------------------------------------------------------------
# Sequence read-out
# --------------------------------------------------------------------------


def _three_to_one_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """``(protein, dna, rna)`` residue-name -> one-letter maps.

    The protein map is ESM's own ``restype_3to1`` because ESMFold2 is the
    consumer; using a different table here would put this adapter and the model
    in disagreement about what a residue is called.
    """
    from esm.utils import residue_constants

    protein = dict(residue_constants.restype_3to1)
    dna = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U", "DI": "I"}
    rna = {"A": "A", "C": "C", "G": "G", "U": "U", "I": "I"}
    return protein, dna, rna


def _ins_codes_at(atoms: AtomArray, starts: np.ndarray) -> tuple[str, ...]:
    """Insertion codes of the residues beginning at *starts*, or blanks."""
    if "ins_code" not in set(atoms.get_annotation_categories()):
        return tuple("" for _ in starts)
    codes = np.asarray(atoms.get_annotation("ins_code")).astype(str)
    return tuple(str(codes[i]) for i in starts)


def _residue_starts(atoms: AtomArray) -> np.ndarray:
    """Indices of the first atom of each residue, in file order."""
    import biotite.structure as struc

    return struc.get_residue_starts(atoms)


def sequence_of_chain(
    atoms: AtomArray,
    kind: str,
    *,
    unknown: str = "X",
) -> tuple[str, list[str]]:
    """One-letter sequence of a polymer chain, plus the residues it could not map.

    This reads only the residues that are **present as atoms**. A structure with
    an unmodelled loop therefore yields a sequence with the loop missing, which
    is a different protein from the one that was crystallized. When the caller
    has ``chain_info`` from ``atomworks.io.parse``, prefer
    ``processed_entity_canonical_sequence`` -- see
    :func:`atom_array_to_structure_prediction_input`, which does exactly that.
    """
    protein_map, dna_map, rna_map = _three_to_one_maps()
    table = {"protein": protein_map, "dna": dna_map, "rna": rna_map}[kind]

    starts = _residue_starts(atoms)
    names = [str(n) for n in atoms.res_name[starts]]
    letters: list[str] = []
    unmapped: list[str] = []
    for name in names:
        letter = table.get(name)
        if letter is None:
            letters.append(unknown)
            unmapped.append(name)
        else:
            letters.append(letter)
    return "".join(letters), unmapped


def _chain_info_entry(chain_info: dict | None, chain_id: str) -> dict | None:
    """The ``chain_info`` record for *chain_id*.

    ``atomworks.io.parse`` keys this dict with ``numpy.str_``, which does not
    hash equal to a plain ``str`` in every numpy version, so look both up.
    """
    if not chain_info:
        return None
    entry = chain_info.get(chain_id)
    if entry is not None:
        return entry
    for key, value in chain_info.items():
        if str(key) == chain_id:
            return value
    return None


def _canonical_sequence_from_chain_info(
    chain_info: dict | None, chain_id: str
) -> str | None:
    """``processed_entity_canonical_sequence`` for *chain_id*, if available.

    This is the sequence including residues that were never resolved, which is
    what should actually be folded.
    """
    entry = _chain_info_entry(chain_info, chain_id)
    if entry is None:
        return None
    sequence = entry.get("processed_entity_canonical_sequence")
    if isinstance(sequence, str) and sequence:
        return sequence
    return None


def _residue_names_from_chain_info(
    chain_info: dict | None, chain_id: str
) -> list[str] | None:
    """The chain's residue names, aligned 1:1 with its canonical sequence.

    Verified on ``1a8o_modified``: ``chain_info["A"]["res_name"]`` has the same
    length as ``processed_entity_canonical_sequence`` (70), and the four ``MSE``
    entries sit at the indices whose canonical letter is ``M``. That alignment
    is what makes a ``Modification`` position meaningful.
    """
    entry = _chain_info_entry(chain_info, chain_id)
    if entry is None:
        return None
    names = entry.get("res_name")
    if names is None:
        return None
    sequence = entry.get("processed_entity_canonical_sequence")
    names = [str(n) for n in names]
    if isinstance(sequence, str) and len(sequence) != len(names):
        # Without a 1:1 alignment a position index would be a guess, and a
        # misplaced modification silently folds the wrong chemistry.
        return None
    return names


def modifications_for_chain(residue_names: list[str], kind: str) -> list[Any]:
    """``Modification`` entries for residues ESM's alphabet cannot express.

    ESMFold2's one-letter sequence covers the standard residues only. Anything
    else -- selenomethionine, a D-amino acid, a phosphorylated serine -- has to
    be named by its CCD code, or the model folds the *parent* residue instead.
    That substitution is silent and produces a perfectly plausible structure of
    a molecule the caller did not ask for.

    Declaring the modification changes tokenization: ``tokenize_protein`` emits
    one token per atom for a modified residue instead of one per residue. That
    is the intended behaviour -- it is how the real chemistry enters the model
    -- but it does mean the token count differs from the unmodified reading.
    """
    from esm.models.esmfold2.types import Modification

    protein_map, dna_map, rna_map = _three_to_one_maps()
    table = {"protein": protein_map, "dna": dna_map, "rna": rna_map}[kind]
    return [
        Modification(position=index, ccd=name)
        for index, name in enumerate(residue_names)
        if name not in table
    ]


# --------------------------------------------------------------------------
# Chain partition
# --------------------------------------------------------------------------
# ESMFold2 takes one input per chain, so a chain is classified as a whole, and
# everything below relies on one label of `chain_key` being one molecule.
# `atomworks.io.parse` makes it so: it gives the polymer and non-polymer
# residues of an author chain chains of their own. An author chain read any
# other way need not be one molecule -- in an mmCIF read with author fields, a
# protein, its ligands and its waters commonly share one. Where the annotations
# can show a chain is mixed, it is refused; where a declaration is all there
# is, the declaration is verified against the CCD instead.


#: Chain kinds a caller may declare. ``unsupported`` is absent: it names the
#: absence of anything to classify by, not something a chain can be. ``water``
#: is present because declaring it describes the chain and asks for nothing;
#: what happens to water is decided by ``drop_water``, as for a parsed one.
DECLARABLE_CHAIN_KINDS: frozenset[str] = frozenset(
    {"protein", "dna", "rna", "ligand", "water"}
)

_POLYMER_KINDS = ("protein", "dna", "rna")

#: The CCD's ``_chem_comp.type`` values for each polymer kind: the partition
#: biotite's ``amino_acid_names`` and ``nucleotide_names`` are drawn from.
_POLYMER_LINK_TYPES: dict[str, frozenset[str]] = {
    "protein": frozenset(
        {
            "D-BETA-PEPTIDE, C-GAMMA LINKING",
            "D-GAMMA-PEPTIDE, C-DELTA LINKING",
            "D-PEPTIDE COOH CARBOXY TERMINUS",
            "D-PEPTIDE LINKING",
            "D-PEPTIDE NH3 AMINO TERMINUS",
            "L-BETA-PEPTIDE, C-GAMMA LINKING",
            "L-GAMMA-PEPTIDE, C-DELTA LINKING",
            "L-PEPTIDE COOH CARBOXY TERMINUS",
            "L-PEPTIDE LINKING",
            "L-PEPTIDE NH3 AMINO TERMINUS",
            "PEPTIDE LINKING",
        }
    ),
    "dna": frozenset(
        {
            "DNA LINKING",
            "DNA OH 3 PRIME TERMINUS",
            "DNA OH 5 PRIME TERMINUS",
            "L-DNA LINKING",
        }
    ),
    "rna": frozenset(
        {
            "L-RNA LINKING",
            "RNA LINKING",
            "RNA OH 3 PRIME TERMINUS",
            "RNA OH 5 PRIME TERMINUS",
        }
    ),
}


def _validated_chain_kinds(
    chain_kinds: Mapping[Any, str] | None, labels: np.ndarray
) -> dict[str, str]:
    """*chain_kinds* keyed by ``str`` chain id, every entry checked to apply.

    A declaration that cannot apply is an error, never a no-op: the caller
    believes something was stated that would otherwise be silently ignored --
    the rule the sequence, MSA and ligand declarations follow.
    """
    declared = _by_chain("chain_kinds", chain_kinds)
    present = sorted(set(labels.tolist()))
    for chain, kind in declared.items():
        if chain not in present:
            raise ChainDeclarationError(
                f"a chain kind is declared for chain {chain!r}, but the structure "
                f"has no such chain (it has {', '.join(map(repr, present))}), so it "
                "would be silently ignored."
            )
        if kind not in DECLARABLE_CHAIN_KINDS:
            raise ChainDeclarationError(
                f"chain {chain!r} is declared {kind!r}; declarable kinds are "
                f"{sorted(DECLARABLE_CHAIN_KINDS)}. 'unsupported' names the absence "
                "of anything to classify a chain by, so it cannot be declared."
            )
    return declared


def _one_per_chain(chain_id: str, annotation: str, values: np.ndarray) -> Any:
    """The one value *annotation* takes on a chain, or ``MixedChainError``.

    Reading the first atom's value and applying it to the chain would fold
    whatever else shares the label as part of that molecule.
    """
    unique = np.unique(values)
    if unique.size == 1:
        return unique[0]
    if annotation == "chain_type":
        from atomworks.enums import ChainType

        def name(value: Any) -> str:
            try:
                return ChainType(int(value)).name
            except ValueError:
                return str(value)

        shown = " and ".join(name(value) for value in unique)
    else:
        shown = " and ".join(str(value) for value in unique)
    raise MixedChainError(
        f"chain {chain_id!r} holds more than one kind of molecule: its atoms "
        f"carry {annotation} {shown}. ESMFold2 takes one input per chain, so "
        "classifying it by any one of them would fold the others as part of that "
        "molecule, or not at all. Give each molecule a chain of its own -- "
        "atomworks.io.parse does -- or pass chain_key= naming an annotation that "
        "does."
    )


def _ccd_class(res_name: str) -> tuple[str, str]:
    """``(class, what the CCD calls it)`` for one residue name.

    The class is a polymer kind, ``"water"``, ``"non-polymer"`` for anything
    else the CCD knows, or ``"unknown"``. Water is AtomWorks' own set
    (``WATER_LIKE_CCDS``, what ``parse`` removes), so a declared water chain
    means what a parsed one does.
    """
    from atomworks.constants import WATER_LIKE_CCDS
    from biotite.structure.info import link_type

    if res_name in WATER_LIKE_CCDS:
        return "water", "water"
    link = link_type(res_name)
    if link is None:
        return "unknown", "not in the CCD"
    link = link.upper()
    for kind, types in _POLYMER_LINK_TYPES.items():
        if link in types:
            return kind, link
    return "non-polymer", link


#: What every residue of a chain declared as each kind must be, for the error.
_DECLARED_RESIDUES_MUST_BE = {
    "protein": "protein residues in the CCD",
    "dna": "DNA residues in the CCD",
    "rna": "RNA residues in the CCD",
    "water": "water",
    "ligand": (
        "possible in a ligand chain, which holds no water, and a polymer "
        "residue only on its own"
    ),
}


def _verify_declared_composition(chain: AtomArray, chain_id: str, kind: str) -> None:
    """Refuse a declared kind that the chain's residues contradict.

    Checked against the CCD because the chain carries nothing else to check it
    against. The CCD is only ever used to refuse: it never supplies a kind, so
    this is verification rather than the residue-name guess that chain
    classification exists to avoid.

    A declaration describes one molecule, so it has to be true of every
    residue. The failure this catches is a chain holding more than one: an
    author chain declared ``protein`` that also holds a heme and a hundred
    waters would fold them as residues of the protein -- or, under a sequence
    override, leave them out altogether -- and nothing downstream raises.
    """
    starts = _residue_starts(chain)
    names = [str(name) for name in chain.res_name[starts]]
    classes = {name: _ccd_class(name) for name in set(names)}

    def fits(name: str) -> bool:
        cls, _ = classes[name]
        if kind in _POLYMER_KINDS:
            return cls == kind
        if kind == "water":
            return cls == "water"
        # A ligand. Never water; and a polymer residue only on its own, since a
        # free amino acid is a ligand but a run of them is a peptide.
        if cls == "water":
            return False
        return cls not in _POLYMER_KINDS or len(names) == 1

    misfits = Counter(name for name in names if not fits(name))
    if not misfits:
        return

    def describe(name: str, count: int) -> str:
        label = f"{count} x {name}" if count > 1 else name
        return f"{label} ({classes[name][1]})"

    listing = ", ".join(
        describe(name, count)
        for name, count in sorted(misfits.items(), key=lambda item: (-item[1], item[0]))
    )
    raise ChainDeclarationError(
        f"chain {chain_id!r} is declared {kind!r}, but {sum(misfits.values())} of "
        f"its {len(names)} residues are not {_DECLARED_RESIDUES_MUST_BE[kind]}: "
        f"{listing}. A declaration describes one molecule, so it has to be true "
        "of every residue. If these are separate molecules -- a ligand, an ion or "
        "water sharing an author chain id -- give each a chain of its own: "
        "atomworks.io.parse does, as does biotite's pdbx.get_structure(..., "
        "use_author_fields=False), or pass chain_key= naming an annotation that "
        "does. If they are this chain's own residues under names the CCD uses "
        "for something else, rename them to their CCD codes."
    )


def chain_records(
    atoms: AtomArray,
    *,
    chain_info: dict | None = None,
    chain_kinds: Mapping[str, str] | None = None,
    chain_key: str = "chain_id",
) -> list[ChainRecord]:
    """Classify every chain of *atoms*, in first-appearance order.

    Order matters: ESMFold2 numbers entities in the order the inputs are given
    (``build_chains_from_input``), so a stable, structure-derived order is what
    makes two conversions of the same structure comparable.

    Args:
        chain_kinds: what each chain is, for an array that does not say -- one
            of :data:`DECLARABLE_CHAIN_KINDS` per chain. A **declaration, not an
            override**, and verified: against ``chain_type`` or ``is_polymer``
            where the array carries them, and against the CCD, residue by
            residue, where it carries neither. It has to be true of every
            residue, so a chain holding more than one molecule cannot be
            declared at all.

            This is the seam for a caller that knows -- a structure built by
            hand, converted back from another modelling package, or threaded
            onto a backbone, none of which carry AtomWorks annotations. The only
            alternative is ``allow_inferred_chain_kind``, which accepts a guess
            that would call a DNA chain protein.

    Raises:
        MixedChainError: a chain whose atoms carry more than one ``chain_type``,
            or more than one ``is_polymer`` value.
        ChainDeclarationError: a ``chain_kinds`` entry that names no chain,
            gives a kind that cannot be declared, or that the chain's
            annotations or residues contradict.
    """
    protein_t, dna_t, rna_t, ligand_t = _chain_type_groups()
    water_t = _droppable_chain_types()

    labels = np.asarray(atoms.get_annotation(chain_key)).astype(str)
    declared_kinds = _validated_chain_kinds(chain_kinds, labels)
    categories = set(atoms.get_annotation_categories())
    chain_types = (
        np.asarray(atoms.get_annotation("chain_type")).astype(int)
        if "chain_type" in categories
        else None
    )
    is_polymer = (
        np.asarray(atoms.get_annotation("is_polymer")).astype(bool)
        if "is_polymer" in categories
        else None
    )

    # First-appearance order, not np.unique's lexicographic order.
    _, first_index = np.unique(labels, return_index=True)
    ordered = [str(labels[i]) for i in sorted(first_index)]

    records: list[ChainRecord] = []
    for chain_id in ordered:
        mask = labels == chain_id
        chain = atoms[mask]
        ctype = (
            int(_one_per_chain(chain_id, "chain_type", chain_types[mask]))
            if chain_types is not None
            else None
        )
        polymer = (
            bool(_one_per_chain(chain_id, "is_polymer", is_polymer[mask]))
            if is_polymer is not None
            else None
        )
        declared_kind = declared_kinds.get(chain_id)

        inferred = False
        if ctype is not None:
            if ctype in protein_t:
                kind = "protein"
            elif ctype in dna_t:
                kind = "dna"
            elif ctype in rna_t:
                kind = "rna"
            elif ctype in ligand_t:
                kind = "ligand"
            elif ctype in water_t:
                kind = "water"
            else:
                kind = "unsupported"
            if declared_kind is not None and declared_kind != kind:
                raise ChainDeclarationError(
                    f"chain {chain_id!r} is declared {declared_kind!r}, but the "
                    f"structure's own chain_type says {kind!r}. A declaration "
                    "states what a chain is; it does not override what the parse "
                    "recorded. Drop the declaration, or fix the annotation."
                )
        elif declared_kind is not None:
            # The caller stated it; what the array still carries has to agree,
            # and the residues have to be what was stated. Not an inference, so
            # `allow_inferred_chain_kind` is never asked for.
            if polymer is not None and polymer != (declared_kind in _POLYMER_KINDS):
                raise ChainDeclarationError(
                    f"chain {chain_id!r} is declared {declared_kind!r}, but "
                    f"is_polymer marks its atoms "
                    f"{'polymer' if polymer else 'non-polymer'}. A declaration "
                    "states what a chain is; it does not override the array's own "
                    "annotation. Drop the declaration, or fix the annotation."
                )
            _verify_declared_composition(chain, chain_id, declared_kind)
            kind = declared_kind
        elif polymer is not None:
            # No chain_type and no declaration: the array did not come through
            # atomworks.io.parse. The polymer branch is a genuine guess -- a DNA
            # or RNA chain arriving this way would be called protein. It is made
            # so that a hand-built AtomArray can still be described, and flagged
            # (`ChainRecord.kind_is_inferred`) so that the adapter refuses to
            # fold on it unless the caller accepts it by name
            # (`allow_inferred_chain_kind`). A caller that knows the answer
            # declares it with `chain_kinds` instead.
            inferred = True
            kind = "protein" if polymer else "ligand"
        else:
            # Nothing to classify by, and no guess made from residue names.
            kind = "unsupported"

        starts = _residue_starts(chain)
        sequence: str | None = None
        if kind in _POLYMER_KINDS:
            canonical = _canonical_sequence_from_chain_info(chain_info, chain_id)
            sequence = (
                canonical
                if canonical is not None
                else sequence_of_chain(chain, kind)[0]
            )

        records.append(
            ChainRecord(
                chain_id=chain_id,
                kind=kind,
                n_atoms=int(mask.sum()),
                n_residues=len(starts),
                sequence=sequence,
                residue_names=tuple(str(n) for n in chain.res_name[starts]),
                residue_ids=tuple(int(r) for r in chain.res_id[starts]),
                ins_codes=_ins_codes_at(chain, starts),
                chain_type=ctype,
                kind_is_inferred=inferred,
            )
        )
    return records


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


def atom_array_to_structure_prediction_input(
    atoms: AtomArray,
    *,
    chain_info: dict | None = None,
    chain_kinds: Mapping[str, str] | None = None,
    ligands: dict[str, LigandSpec] | tuple[LigandSpec, ...] = (),
    msas: dict[str, Any] | None = None,
    sequences: dict[str, str] | None = None,
    allow_undeclared_ccd_ligands: bool = True,
    emit_modifications: bool = True,
    declare_covalent_bonds: bool = True,
    allow_unresolved_covalent_bonds: bool = False,
    allow_unsupported_chains: bool = False,
    allow_inferred_chain_kind: bool = False,
    allow_unplaceable_modifications: bool = False,
    drop_water: bool = True,
    chain_key: str = "chain_id",
    report: AdapterReport | None = None,
) -> Any:
    """Convert an AtomWorks ``AtomArray`` into an ESMFold2 input.

    Args:
        atoms: the structure, as returned by ``atomworks.io.parse``.
        chain_info: the ``chain_info`` block from the same parse. Strongly
            preferred: it carries the full entity sequence, including residues
            that were never resolved. Without it, unmodelled residues are
            simply absent from the sequence, which folds a different molecule.
        chain_kinds: what each chain is, for a structure that carries no
            ``chain_type`` -- one built by hand, converted back from another
            modelling package, or threaded onto a backbone. A declaration,
            verified rather than trusted, never an override; see
            :func:`chain_records`. Each chain has to be one molecule, which a
            raw author chain often is not.
        ligands: declared identities for non-polymer chains, keyed by chain id
            (or a tuple, keyed by each spec's own ``chain_id``). A key must
            equal its spec's ``chain_id``, and a chain has at most one spec.
        msas: per-chain ``MSA`` objects for protein chains, each with the
            sequence being folded there as its query row.
        sequences: per-chain sequences that override what the structure says.
            The sequence-override path: fold *this* sequence on *that* system.
        allow_undeclared_ccd_ligands: when a non-polymer chain has no declared
            spec, use its residue name as a CCD code. Names that carry no
            chemical meaning (``LIG``, ``UNL``, ``UNK``) are refused regardless,
            because those are exactly the labels that collide.
        allow_unresolved_covalent_bonds: proceed when a bond in the source
            cannot be placed in the model's indexing. Off by default: the
            alternative to raising is folding a connected system as though it
            were disconnected, which nothing downstream can detect.
        allow_unsupported_chains: proceed when a chain cannot be expressed as
            any ESMFold2 input. Off by default for the same reason -- the
            direct API returns no report, so the chain would simply be absent
            from a confident prediction. Water is unaffected; it is dropped
            under ``drop_water``.
        allow_inferred_chain_kind: proceed when a chain has no ``chain_type``
            and its kind must be guessed from ``is_polymer``. Off by default:
            the guess cannot tell protein from nucleic acid, and a DNA chain
            guessed this way is folded as a poly-X protein.
        allow_unplaceable_modifications: proceed when a non-standard residue
            cannot be tied to a sequence position, folding its parent residue
            instead. Off by default, because that is a different molecule.
        emit_modifications: declare non-standard polymer residues by CCD code
            rather than folding the parent residue. See
            :func:`modifications_for_chain` for what this changes.
        drop_water: drop water chains, recording them in the report.
        report: filled in with what happened, if given.

    Returns:
        ``StructurePredictionInput`` ready for ``ESMFold2InputBuilder``.

    Raises:
        ChainDeclarationError: a chain kind, sequence override, MSA or ligand
            spec that would not reach exactly the chain it names, or that the
            chain contradicts.
        MixedChainError: a chain whose atoms belong to more than one kind of
            molecule, by their own annotations.
        LigandIdentityError: a non-polymer chain whose identity cannot be
            established without guessing.

    Every degradation in :data:`DEGRADATIONS` also raises its own error unless
    it is accepted by name.
    """
    from esm.models.esmfold2.types import (
        DNAInput,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    rep = report if report is not None else AdapterReport()
    spec_by_chain = _index_ligand_specs(ligands)
    msas = _by_chain("msas", msas)
    sequences = _by_chain("sequences", sequences)

    records = chain_records(
        atoms, chain_info=chain_info, chain_kinds=chain_kinds, chain_key=chain_key
    )
    rep.chains = records
    _check_declarations(records, sequences=sequences, msas=msas, ligands=spec_by_chain)

    labels = np.asarray(atoms.get_annotation(chain_key)).astype(str)
    inputs: list[Any] = []

    for record in records:
        chain_id = record.chain_id

        if record.kind_is_inferred:
            if not allow_inferred_chain_kind:
                raise InferredChainKindError(
                    f"chain {chain_id!r} has no chain_type annotation, so its kind "
                    f"would be inferred from is_polymer as {record.kind!r}. That "
                    "cannot tell protein from DNA or RNA -- a nucleic-acid chain "
                    "inferred this way is folded as a protein -- nor water from a "
                    "ligand. "
                    + _declare_hint(chain_id)
                    + " To accept the guess instead: "
                    + _remedy("inferred_chain_kind")
                )
            rep.inferred_chain_kinds.append(chain_id)

        if record.kind == "water":
            if drop_water:
                rep.dropped.append((chain_id, "water"))
                continue
            raise ValueError(
                f"chain {chain_id!r} is water and ESMFold2 has no water input; "
                "pass drop_water=True to drop it explicitly"
            )

        if record.kind == "unsupported":
            reason = (
                f"unsupported chain_type {record.chain_type!r}"
                if record.chain_type is not None
                else "no chain_type or is_polymer annotation"
            )
            if not allow_unsupported_chains:
                # With no annotation at all the chain may be perfectly
                # expressible; what is missing is someone saying what it is. A
                # chain_type upstream has no input for cannot be declared away.
                declare = (
                    _declare_hint(chain_id) + " To drop it instead: "
                    if record.chain_type is None
                    else ""
                )
                raise UnsupportedChainError(
                    f"chain {chain_id!r} cannot be expressed as an ESMFold2 input "
                    f"({reason}), so folding would silently omit it. "
                    + declare
                    + _remedy("unsupported_chains")
                )
            rep.dropped.append((chain_id, reason))
            continue

        if record.is_polymer:
            sequence = sequences.get(chain_id, record.sequence or "")
            if not sequence:
                # An empty override is refused with the other declarations, and
                # a chain with residues always yields a sequence; kept so that
                # an empty sequence can never quietly drop a chain.
                raise ValueError(
                    f"chain {chain_id!r} has an empty sequence; folding would omit it"
                )

            rep.sequence_source[chain_id] = (
                "override"
                if chain_id in sequences
                else (
                    "chain_info"
                    if _canonical_sequence_from_chain_info(chain_info, chain_id)
                    else "atoms"
                )
            )
            _, unmapped = sequence_of_chain(atoms[labels == chain_id], record.kind)
            if unmapped:
                rep.unknown_residues[chain_id] = unmapped

            mods = _modifications_for(
                record,
                chain_id,
                chain_info=chain_info,
                sequence=sequence,
                overridden=chain_id in sequences,
                emit=emit_modifications,
                report=rep,
                allow_unplaceable=allow_unplaceable_modifications,
            )

            if record.kind == "protein":
                inputs.append(
                    ProteinInput(
                        id=chain_id,
                        sequence=sequence,
                        msa=msas.get(chain_id),
                        modifications=mods or None,
                    )
                )
            elif record.kind == "dna":
                inputs.append(
                    DNAInput(id=chain_id, sequence=sequence, modifications=mods or None)
                )
            else:
                inputs.append(
                    RNAInput(id=chain_id, sequence=sequence, modifications=mods or None)
                )
            continue

        # Non-polymer.
        chain_atoms = atoms[labels == chain_id]
        spec = spec_by_chain.get(chain_id)
        if spec is None:
            spec = _spec_from_ccd_annotation(
                record, chain_id, allow=allow_undeclared_ccd_ligands
            )
        spec.verify_against(chain_atoms)
        inputs.append(spec.to_esm(chain_id))

    if not inputs:
        raise ValueError(
            "no ESMFold2 inputs were produced from this structure "
            f"({rep.dropped_summary()})"
        )

    spi = StructurePredictionInput(sequences=inputs)
    if declare_covalent_bonds:
        spi = _attach_covalent_bonds(
            spi,
            atoms,
            records=records,
            chain_key=chain_key,
            chain_info=chain_info,
            report=rep,
            allow_unresolved=allow_unresolved_covalent_bonds,
        )
    return spi


def _attach_covalent_bonds(
    spi: Any,
    atoms: AtomArray,
    *,
    records: list[ChainRecord],
    chain_key: str,
    chain_info: dict | None,
    report: AdapterReport,
    allow_unresolved: bool = False,
) -> Any:
    """Return *spi* with any non-inferable covalent bonds declared.

    Short-circuits before doing any work when the structure has no such bond,
    which is the common case: resolving them costs one extra featurization,
    because the atom indices ESM wants are positions in the tokenizer's own
    per-residue ordering (see :mod:`esmfold2_atomworks.data.bonds`).
    """
    from esm.models.esmfold2.types import StructurePredictionInput

    from esmfold2_atomworks.data.bonds import (
        covalent_bond_candidates,
        resolve_covalent_bonds,
    )

    # Water is omitted under an explicit policy, so its bonds are genuinely
    # uninteresting. Every other chain that failed to reach the model is a
    # different matter: a bond touching it is real, unplaceable, and exactly
    # what the caller's strictness policy is for -- so it is partitioned out
    # below rather than filtered away here, where it would escape that policy.
    water = frozenset(record.chain_id for record in records if record.kind == "water")
    represented = frozenset(str(entry.id) for entry in spi.sequences)

    candidates = covalent_bond_candidates(
        atoms, chain_key=chain_key, ignore_chains=water
    )
    if not candidates:
        return spi

    placeable, unrepresented = [], []
    for candidate in candidates:
        missing = sorted(
            str(chain) for chain in {candidate.chain_1, candidate.chain_2} - represented
        )
        if missing:
            unrepresented.append(
                f"{candidate.describe()}: endpoint chain(s) {missing} are not "
                "represented in the model input"
            )
        else:
            placeable.append(candidate)

    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    features, chain_infos = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=0)
    bonds, skipped = resolve_covalent_bonds(
        placeable,
        features,
        chain_infos,
        _residue_index_map(records, chain_info, report),
    )
    skipped = unrepresented + skipped
    report.covalent_bonds = list(candidates)
    if skipped and not allow_unresolved:
        raise CovalentBondResolutionError(
            f"{len(skipped)} covalent bond(s) in the source could not be placed "
            "in the model's indexing, so folding would treat a connected system "
            "as disconnected:\n  "
            + "\n  ".join(skipped)
            + "\n"
            + _remedy("unresolved_covalent_bonds")
        )
    report.unresolved_covalent_bonds = skipped
    if not bonds:
        return spi
    return StructurePredictionInput(
        sequences=list(spi.sequences),
        distogram_conditioning=spi.distogram_conditioning,
        covalent_bonds=bonds,
    )


def _residue_index_map(
    records: list[ChainRecord],
    chain_info: dict | None,
    report: AdapterReport | None = None,
) -> dict[tuple[str, int, str], int]:
    """``(chain_id, res_id, ins_code) -> tokenizer residue index``.

    The deposited numbering need not start at one, need not be contiguous, and
    may repeat a number under different insertion codes -- ``100``, ``100A``,
    ``100B`` -- so it is only related to the model's indices through the residue
    list the sequence was built from.

    Two sources, and which one is usable depends on the chain:

    * ``chain_info`` lists every residue of the entity, including those never
      resolved, which is what makes it the right basis for the sequence. But it
      carries **no insertion code**, so for a chain that uses them it cannot say
      which of ``100``/``100A`` a given entry is.
    * The observed residues carry both, but omit anything unresolved.

    So when the sequence came from ``chain_info`` *and* the chain uses insertion
    codes, the two cannot be reconciled here. That chain is left out of the map
    and named in the report, rather than silently attached to whichever residue
    happened to overwrite the others: a bond on it then fails to resolve --
    which raises unless unresolved bonds are accepted -- and its labels are
    skipped with a reason.
    """
    mapping: dict[tuple[str, int, str], int] = {}
    for record in records:
        observed_uses_ins_codes = any(code for code in record.ins_codes)

        entry = _chain_info_entry(chain_info, record.chain_id)
        res_ids = entry.get("res_id") if entry else None

        if res_ids is not None and observed_uses_ins_codes:
            if report is not None:
                report.unrepresentable_insertion_codes.append(record.chain_id)
            continue

        if res_ids is not None:
            for index, res_id in enumerate(res_ids):
                mapping[(record.chain_id, int(res_id), "")] = index
            continue

        # No chain_info: the sequence came from the observed residues, so their
        # order *is* the model's residue order and insertion codes are usable.
        for index, (res_id, ins_code) in enumerate(
            zip(record.residue_ids, record.ins_codes, strict=False)
        ):
            mapping[(record.chain_id, int(res_id), str(ins_code))] = index
    return mapping


def _modifications_for(
    record: ChainRecord,
    chain_id: str,
    *,
    chain_info: dict | None,
    sequence: str,
    overridden: bool,
    emit: bool,
    report: AdapterReport,
    allow_unplaceable: bool = False,
) -> list[Any]:
    """Modifications for one polymer chain, or an empty list with a reason.

    A modification is only emitted when its position is *known* to line up with
    the sequence being folded. An overridden sequence is a
    different molecule from the one in the structure, so carrying the
    structure's modifications onto it would place them by coincidence.
    """
    if not emit or overridden:
        return []

    names = _residue_names_from_chain_info(chain_info, chain_id)
    if names is None:
        names = list(record.residue_names)
        if len(names) != len(sequence):
            # A degradation only if something actually needed placing: a chain
            # of standard residues folds identically with or without this, and
            # raising on it would be a false alarm.
            protein_map, dna_map, rna_map = _three_to_one_maps()
            table = {"protein": protein_map, "dna": dna_map, "rna": rna_map}[
                record.kind
            ]
            nonstandard = sorted({name for name in names if name not in table})
            if nonstandard:
                if not allow_unplaceable:
                    raise ModificationResolutionError(
                        f"chain {chain_id!r} carries non-standard residue(s) "
                        f"{nonstandard}, but its {len(names)} modelled residues "
                        f"cannot be aligned to its {len(sequence)}-residue sequence, "
                        "so their positions are unknown and the parent residues "
                        "would be folded instead. "
                        + _remedy("unplaceable_modifications")
                    )
                report.unplaceable_modifications.append(chain_id)
            return []

    mods = modifications_for_chain(names, record.kind)
    if mods:
        report.modifications[chain_id] = [(m.position, m.ccd) for m in mods]
    return mods


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------


def _index_ligand_specs(
    ligands: Mapping[Any, LigandSpec] | Iterable[LigandSpec] | None,
) -> dict[str, LigandSpec]:
    """``chain_id -> LigandSpec``, refusing a declaration that contradicts itself.

    In the mapping form the key and the spec both name a chain, and they must
    name the same one; otherwise the spec is applied to the key's chain while
    describing another. In either form a chain has one identity, so a second
    spec for it is refused rather than resolved as last-one-wins.
    """
    if not ligands:
        return {}
    if isinstance(ligands, Mapping):
        pairs = [(str(key), spec) for key, spec in ligands.items()]
    else:
        pairs = [(None, spec) for spec in ligands]

    index: dict[str, LigandSpec] = {}
    for key, spec in pairs:
        if not isinstance(spec, LigandSpec):
            raise TypeError(f"ligands must hold LigandSpec, not {type(spec).__name__}")
        chain = str(spec.chain_id)
        if key is not None and key != chain:
            raise ChainDeclarationError(
                f"ligands[{key!r}] is a LigandSpec for chain {chain!r}. The key and "
                "the spec must name the same chain, or the spec would be applied "
                "to one chain while describing another."
            )
        if chain in index:
            raise ChainDeclarationError(
                f"chain {chain!r} is declared by two LigandSpecs. A chain has one "
                "identity, and the second would silently replace the first."
            )
        index[chain] = spec
    return index


def _by_chain(what: str, declared: Mapping[Any, Any] | None) -> dict[str, Any]:
    """*declared* keyed by ``str`` chain id, as the structure's own labels are.

    A key that is not a string -- ``1`` for chain ``"1"``, as a config file can
    produce -- would otherwise match no chain and be ignored.
    """
    if not declared:
        return {}
    by_chain: dict[str, Any] = {}
    for key, value in declared.items():
        chain = str(key)
        if chain in by_chain:
            raise ChainDeclarationError(f"{what} declares chain {chain!r} twice")
        by_chain[chain] = value
    return by_chain


def _check_declarations(
    records: list[ChainRecord],
    *,
    sequences: dict[str, Any],
    msas: dict[str, Any],
    ligands: dict[str, LigandSpec],
) -> None:
    """Refuse any declaration that would not reach the chain it names.

    Each of *sequences*, *msas* and *ligands* is a statement about one chain,
    and each is read only by chains of particular kinds. Anything else is
    passed over by the conversion without a trace: an override for a chain
    that is not there folds the structure's own sequence, an MSA on a
    nucleic-acid chain leaves the protein it was meant for in single-sequence
    mode, a ligand identity on a polymer chain is never read. Checked before
    anything is converted, so the error names the declaration rather than a
    symptom of it.
    """
    by_id = {record.chain_id: record for record in records}

    for chain, sequence in sequences.items():
        record = _bind(
            by_id,
            chain,
            "a sequence override",
            ("protein", "dna", "rna"),
            "a protein, DNA or RNA chain",
        )
        if not sequence:
            raise ChainDeclarationError(
                f"chain {chain!r} has an empty override; folding would omit the "
                "chain. Supply a sequence, or remove the chain from the structure "
                "if it is meant to be absent."
            )
        if record.kind == "protein" and any(mark in sequence for mark in ":|"):
            # clean_esmfold2_input, which every fold passes through, splits a
            # protein sequence at these marks into chains `<id>_0`, `<id>_1`.
            raise ChainDeclarationError(
                f"the sequence override for chain {chain!r} contains a chain break "
                f"(':' or '|'). Upstream splits a protein sequence there into "
                f"separate chains ({chain}_0, {chain}_1, ...), so chain {chain!r} "
                "would no longer be folded as one chain under its own id. An "
                "override describes one chain; several chains must be separate "
                "chains of the structure."
            )

    for chain, msa in msas.items():
        record = _bind(
            by_id,
            chain,
            "an MSA",
            ("protein",),
            "a protein chain (ESMFold2 reads MSAs for protein chains only)",
        )
        if msa is not None:
            _check_msa_binds(chain, msa, sequences.get(chain) or record.sequence or "")

    for chain in ligands:
        _bind(by_id, chain, "a LigandSpec", ("ligand",), "a non-polymer chain")


def _bind(
    by_id: dict[str, ChainRecord],
    chain: str,
    what: str,
    kinds: tuple[str, ...],
    applies_to: str,
) -> ChainRecord:
    record = by_id.get(chain)
    if record is None:
        present = ", ".join(repr(chain_id) for chain_id in by_id)
        raise ChainDeclarationError(
            f"{what} is declared for chain {chain!r}, but the structure has no such "
            f"chain (it has {present}), so it would be silently ignored."
        )
    if record.kind not in kinds:
        inferred = (
            " (inferred from is_polymer, for want of a chain_type)"
            if record.kind_is_inferred
            else ""
        )
        raise ChainDeclarationError(
            f"{what} is declared for chain {chain!r}, whose kind is "
            f"{record.kind!r}{inferred}; it applies only to {applies_to}, so it "
            "would be silently ignored."
        )
    return record


def _check_msa_binds(chain_id: str, msa: Any, sequence: str) -> None:
    """An MSA binds to a chain only if its query row is the sequence folded there.

    Upstream does not check. ``construct_paired_msa`` clamps each residue's
    column to the alignment's width, so an alignment built for another sequence
    -- the other chain of a heteromer, a construct with a different tag, the
    wild type of a variant -- is used anyway, column by column, and its first row
    contradicts the residues the model is given.
    """
    query = getattr(msa, "query", None)
    if not isinstance(query, str):
        raise TypeError(
            f"the MSA for chain {chain_id!r} is a {type(msa).__name__}, not an "
            "esm.utils.msa.MSA"
        )
    # Aligned columns only: a3m insertions (lowercase, '.') occupy no column.
    aligned = "".join(ch for ch in query if not (ch == "." or ch.islower()))
    if aligned == sequence:
        return

    if len(aligned) != len(sequence):
        consequence = (
            "every residue past its end would read its last column"
            if len(aligned) < len(sequence)
            else "its trailing columns would be dropped"
        )
        detail = (
            f"it has {len(aligned)} aligned columns for a {len(sequence)}-residue "
            f"chain, and ESMFold2 clamps each residue to the alignment's width, so "
            f"{consequence}"
        )
    else:
        differ = [i for i, (a, b) in enumerate(zip(aligned, sequence)) if a != b]
        first = differ[0]
        detail = (
            f"its query row differs from the folded sequence at {len(differ)} of "
            f"{len(sequence)} positions (first at {first}: {aligned[first]!r} vs "
            f"{sequence[first]!r})"
        )
    raise ChainDeclarationError(
        f"the MSA for chain {chain_id!r} is aligned to a different sequence than "
        f"the one being folded: {detail}. Build it with the folded sequence as its "
        "query row; to fold a variant against another sequence's alignment, "
        "replace the query row with the variant."
    )


def _spec_from_ccd_annotation(
    record: ChainRecord, chain_id: str, *, allow: bool
) -> LigandSpec:
    """Build a spec from the chain's residue names, or refuse to guess.

    A CCD code that came through ``atomworks.io.parse`` *has* been reconciled
    against the component dictionary, so it is a real statement about chemistry
    -- unlike a name a model wrote on its own output. The generic labels are
    refused because they are precisely the ones where the two cases are
    indistinguishable from the string alone.
    """
    names = [n for n in record.residue_names if n]
    if not names:
        raise LigandIdentityError(f"non-polymer chain {chain_id!r} has no residues")

    generic = sorted({n for n in names if n in GENERIC_LIGAND_NAMES})
    if generic:
        raise LigandIdentityError(
            f"non-polymer chain {chain_id!r} is labelled {generic}, which carries no "
            "chemical meaning even though each is a real CCD code. Declare it with "
            "LigandSpec(chain_id=..., smiles=...) or ccd=..."
        )
    if not allow:
        raise LigandIdentityError(
            f"non-polymer chain {chain_id!r} has no LigandSpec and "
            "allow_undeclared_ccd_ligands=False"
        )

    # A residue name is only usable as a CCD code if it *is* one. Names that
    # are not include AtomWorks' placeholders for a ligand built from SMILES or
    # an SDF (`L:0`, `C:0`) and anything the depositor invented. Left
    # unchecked, those reach the featurizer and fail there with
    # "CCD component L:0 not found", which points at neither the chain nor the
    # fix. Checking here turns "trust the label" into "verify the label",
    # which is what D-004 asks for.
    unknown = sorted(name for name in set(names) if not _is_ccd_code(name))
    if unknown:
        raise LigandIdentityError(
            f"non-polymer chain {chain_id!r} is labelled {unknown}, which is not in "
            "the chemical component dictionary -- so it cannot be used as a CCD "
            "code. A ligand built from SMILES or an SDF carries a placeholder name "
            "of this kind. Declare it with LigandSpec(chain_id=..., smiles=...)."
        )
    return LigandSpec(chain_id=chain_id, ccd=tuple(names))


def _is_ccd_code(name: str) -> bool:
    """Whether *name* names a component in the CCD.

    Returns ``True`` when the dictionary cannot be consulted at all: refusing
    every ligand because an optional asset is missing would be worse than the
    late failure this check exists to improve on.
    """
    try:
        from esm.models.esmfold2.conformers import load_ccd

        from esmfold2_atomworks import paths

        ccd = load_ccd(paths.ccd_dir())
    except Exception:  # noqa: BLE001 - absence of the CCD must not be fatal here
        return True
    try:
        return name in ccd
    except TypeError:
        return True
