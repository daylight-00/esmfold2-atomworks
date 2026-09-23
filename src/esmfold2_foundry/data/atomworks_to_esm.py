"""``AtomWorks AtomArray`` -> ``ESMFold2 StructurePredictionInput``.

This is Phase 1 of the port, and the only thing the first milestone needs: with
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
(``esmfold2_foundry.parity``).

**What is deliberately not inferred.** Chain classification comes from
AtomWorks' own ``chain_type`` annotation, and ligand identity from an explicit
declaration or a CCD code that is verified, never from a residue-name guess.
See :mod:`esmfold2_foundry.data.spec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from esmfold2_foundry.data.spec import (
    GENERIC_LIGAND_NAMES,
    CovalentBondResolutionError,
    LigandIdentityError,
    LigandSpec,
    UnsupportedChainError,
)

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = [
    "AdapterReport",
    "ChainRecord",
    "atom_array_to_structure_prediction_input",
    "chain_records",
    "modifications_for_chain",
    "sequence_of_chain",
]


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
    #: a statement.
    kind_is_inferred: bool = False

    @property
    def is_polymer(self) -> bool:
        return self.kind in ("protein", "dna", "rna")


@dataclass
class AdapterReport:
    """What the adapter did, so that nothing it dropped is invisible.

    A conversion that quietly discards a chain is the failure mode this whole
    project is trying to make impossible: the fold still succeeds, the metrics
    still look reasonable, and the model was given a different system than the
    caller believes. Every chain that does not reach ESMFold2 is named here.
    """

    chains: list[ChainRecord] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (chain_id, reason)
    sequence_source: dict[str, str] = field(default_factory=dict)
    unknown_residues: dict[str, list[str]] = field(default_factory=dict)
    modifications: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    #: Chains carrying non-standard residues whose position could not be tied
    #: to the folded sequence. Folding proceeds with the parent residues, so
    #: this is the one case where chemistry is knowingly approximated.
    unplaceable_modifications: list[str] = field(default_factory=list)
    #: Covalent bonds carried across from the source structure.
    covalent_bonds: list[Any] = field(default_factory=list)
    #: Bonds detected but not placeable in the model's indexing, with the
    #: reason. Skipped rather than guessed -- a wrong index bonds the wrong
    #: pair of atoms, which upstream cannot detect.
    unresolved_covalent_bonds: list[str] = field(default_factory=list)
    #: Chains whose residues carry insertion codes that cannot be tied to
    #: the sequence, because ``chain_info`` does not record them. Their
    #: bonds and labels are skipped rather than mis-assigned.
    unrepresentable_insertion_codes: list[str] = field(default_factory=list)

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


def chain_records(
    atoms: AtomArray,
    *,
    chain_info: dict | None = None,
    chain_key: str = "chain_id",
) -> list[ChainRecord]:
    """Classify every chain of *atoms*, in first-appearance order.

    Order matters: ESMFold2 numbers entities in the order the inputs are given
    (``build_chains_from_input``), so a stable, structure-derived order is what
    makes two conversions of the same structure comparable.
    """
    protein_t, dna_t, rna_t, ligand_t = _chain_type_groups()
    water_t = _droppable_chain_types()

    labels = np.asarray(atoms.get_annotation(chain_key)).astype(str)
    has_chain_type = "chain_type" in set(atoms.get_annotation_categories())
    chain_types = (
        np.asarray(atoms.get_annotation("chain_type")).astype(int)
        if has_chain_type
        else None
    )
    has_is_polymer = "is_polymer" in set(atoms.get_annotation_categories())
    is_polymer = (
        np.asarray(atoms.get_annotation("is_polymer")).astype(bool)
        if has_is_polymer
        else None
    )

    # First-appearance order, not np.unique's lexicographic order.
    _, first_index = np.unique(labels, return_index=True)
    ordered = [labels[i] for i in sorted(first_index)]

    records: list[ChainRecord] = []
    for chain_id in ordered:
        mask = labels == chain_id
        chain = atoms[mask]
        ctype = int(chain_types[mask][0]) if chain_types is not None else None

        if ctype is None:
            # No chain_type annotation: the array did not come through
            # atomworks.io.parse. Fall back to is_polymer, and if that is
            # missing too, say so rather than guessing from residue names.
            #
            # The polymer branch is a genuine guess -- a DNA or RNA chain
            # arriving without chain_type would be called protein here. It is
            # kept because refusing would reject every hand-built AtomArray,
            # but it is recorded (`ChainRecord.kind_is_inferred`) so a caller
            # can tell an inference from a statement. Anything from
            # `atomworks.io.parse` or the component assembler carries
            # chain_type and never takes this path.
            inferred = is_polymer is not None
            if is_polymer is None:
                kind = "unsupported"
            else:
                kind = "protein" if bool(is_polymer[mask][0]) else "ligand"
        elif ctype in protein_t:
            inferred = False
            kind = "protein"
        elif ctype in dna_t:
            inferred = False
            kind = "dna"
        elif ctype in rna_t:
            inferred = False
            kind = "rna"
        elif ctype in ligand_t:
            inferred = False
            kind = "ligand"
        elif ctype in water_t:
            inferred = False
            kind = "water"
        else:
            inferred = False
            kind = "unsupported"

        starts = _residue_starts(chain)
        sequence: str | None = None
        if kind in ("protein", "dna", "rna"):
            declared = _canonical_sequence_from_chain_info(chain_info, str(chain_id))
            sequence = (
                declared if declared is not None else sequence_of_chain(chain, kind)[0]
            )

        records.append(
            ChainRecord(
                chain_id=str(chain_id),
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
    ligands: dict[str, LigandSpec] | tuple[LigandSpec, ...] = (),
    msas: dict[str, Any] | None = None,
    sequences: dict[str, str] | None = None,
    allow_undeclared_ccd_ligands: bool = True,
    emit_modifications: bool = True,
    declare_covalent_bonds: bool = True,
    allow_unresolved_covalent_bonds: bool = False,
    allow_unsupported_chains: bool = False,
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
        ligands: declared identities for non-polymer chains, keyed by chain id
            (or a tuple, keyed by each spec's own ``chain_id``).
        msas: per-chain ``MSA`` objects, attached to the matching protein chain.
        sequences: per-chain sequences that override what the structure says.
            This is the design path -- fold *this* sequence on *that* system.
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
        emit_modifications: declare non-standard polymer residues by CCD code
            rather than folding the parent residue. See
            :func:`modifications_for_chain` for what this changes.
        drop_water: drop water chains, recording them in the report.
        report: filled in with what happened, if given.

    Returns:
        ``StructurePredictionInput`` ready for ``ESMFold2InputBuilder``.

    Raises:
        LigandIdentityError: a non-polymer chain whose identity cannot be
            established without guessing.
    """
    from esm.models.esmfold2.types import (
        DNAInput,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    rep = report if report is not None else AdapterReport()
    spec_by_chain = _index_ligand_specs(ligands)
    msas = msas or {}
    sequences = sequences or {}

    records = chain_records(atoms, chain_info=chain_info, chain_key=chain_key)
    rep.chains = records

    labels = np.asarray(atoms.get_annotation(chain_key)).astype(str)
    inputs: list[Any] = []

    for record in records:
        chain_id = record.chain_id

        if record.kind == "water":
            if drop_water:
                rep.dropped.append((chain_id, "water"))
                continue
            raise ValueError(
                f"chain {chain_id!r} is water and ESMFold2 has no water input; "
                "pass drop_water=True to drop it explicitly"
            )

        if record.kind == "unsupported":
            reason = f"unsupported chain_type {record.chain_type!r}"
            rep.dropped.append((chain_id, reason))
            if not allow_unsupported_chains:
                raise UnsupportedChainError(
                    f"chain {chain_id!r} cannot be expressed as an ESMFold2 input "
                    f"({reason}), so folding would silently omit it. Pass "
                    "allow_unsupported_chains=True to drop it deliberately."
                )
            continue

        if record.is_polymer:
            sequence = sequences.get(chain_id, record.sequence or "")
            if not sequence:
                rep.dropped.append((chain_id, "empty sequence"))
                continue

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
    per-residue ordering (see :mod:`esmfold2_foundry.data.bonds`).
    """
    from esm.models.esmfold2.types import StructurePredictionInput

    from esmfold2_foundry.data.bonds import (
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
    report.unresolved_covalent_bonds = skipped
    if skipped and not allow_unresolved:
        raise CovalentBondResolutionError(
            f"{len(skipped)} covalent bond(s) in the source could not be placed "
            "in the model's indexing, so folding would treat a connected system "
            "as disconnected:\n  "
            + "\n  ".join(skipped)
            + "\nPass allow_unresolved_covalent_bonds=True to proceed anyway."
        )
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
    and named in the report: its bonds and labels are then skipped with a
    reason, rather than silently attached to whichever residue happened to
    overwrite the others.
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
) -> list[Any]:
    """Modifications for one polymer chain, or an empty list with a reason.

    A modification is only emitted when its position is *known* to line up with
    the sequence being folded. An overridden sequence (the design path) is a
    different molecule from the one in the structure, so carrying the
    structure's modifications onto it would place them by coincidence.
    """
    if not emit or overridden:
        return []

    names = _residue_names_from_chain_info(chain_info, chain_id)
    if names is None:
        names = list(record.residue_names)
        if len(names) != len(sequence):
            if record.chain_type is not None:
                report.unplaceable_modifications.append(chain_id)
            return []

    mods = modifications_for_chain(names, record.kind)
    if mods:
        report.modifications[chain_id] = [(m.position, m.ccd) for m in mods]
    return mods


def _index_ligand_specs(
    ligands: dict[str, LigandSpec] | tuple[LigandSpec, ...],
) -> dict[str, LigandSpec]:
    if isinstance(ligands, dict):
        return dict(ligands)
    return {spec.chain_id: spec for spec in ligands}


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

        ccd = load_ccd()
    except Exception:  # noqa: BLE001 - absence of the CCD must not be fatal here
        return True
    try:
        return name in ccd
    except TypeError:
        return True
