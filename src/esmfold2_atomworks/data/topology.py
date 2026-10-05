"""A bond list for a folded structure, from the chemistry the model was given.

ESMFold2's output has no bond field (``MolecularComplex`` records none), so the
structure :func:`~esmfold2_atomworks.data.molecular_complex.molecular_complex_to_atom_array`
returns carries no ``BondList``. The model was nevertheless handed everything that
determines the topology: each chain's kind and residues, each ligand's CCD codes
or SMILES, and the covalent bonds that were declared. :func:`build_bond_list`
rebuilds the bonds from those, and from nothing the model reported:

* inside a residue or CCD ligand component, the CCD's bonds with the types its
  file gives them (``value_order`` and the aromatic flag, so ``AROMATIC_SINGLE``
  and ``AROMATIC_DOUBLE`` in a ring), among the atoms that are present;
* inside a SMILES ligand, RDKit's bonds, with the atom names ESMFold2 gives it;
* between consecutive residues of a polymer, the peptide bond (``C`` to ``N``) or
  the phosphodiester bond (``O3'`` to ``P``);
* the declared covalent bonds, with the order the source structure had.

Every atom the chemistry names has to be found in the structure under the name
the chemistry gives it, and every atom of the structure has to be accounted for;
otherwise :class:`~esmfold2_atomworks.data.spec.TopologyError` is raised. A bond
list that looks complete and is not would be worse than none.

:func:`ccd_name_collisions` is the check on the other side of a written file: a
residue name that is a CCD code for a different molecule is read back from the
dictionary, not from the file.
"""

from __future__ import annotations

from functools import cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from esmfold2_atomworks.data.spec import LigandIdentityError, TopologyError

if TYPE_CHECKING:
    from biotite.structure import AtomArray, BondList

__all__ = ["build_bond_list", "ccd_name_collisions", "smiles_atom_names"]

_POLYMER_INPUTS = {
    "ProteinInput": ("C", "N"),
    "DNAInput": ("O3'", "P"),
    "RNAInput": ("O3'", "P"),
}


def _biotite_bond_type(rdkit_bond: Any) -> int:
    """The ``biotite.structure.BondType`` of an RDKit bond, as an integer."""
    from biotite.structure import BondType
    from rdkit import Chem

    kind = rdkit_bond.GetBondType()
    aromatic = rdkit_bond.GetIsAromatic()
    if kind == Chem.BondType.SINGLE:
        return int(BondType.AROMATIC_SINGLE if aromatic else BondType.SINGLE)
    if kind == Chem.BondType.DOUBLE:
        return int(BondType.AROMATIC_DOUBLE if aromatic else BondType.DOUBLE)
    if kind == Chem.BondType.TRIPLE:
        return int(BondType.AROMATIC_TRIPLE if aromatic else BondType.TRIPLE)
    if kind == Chem.BondType.AROMATIC:
        return int(BondType.AROMATIC)
    if kind in (Chem.BondType.DATIVE, Chem.BondType.DATIVEL, Chem.BondType.DATIVER):
        return int(BondType.COORDINATION)
    return int(BondType.ANY)


@cache
def _ccd_bond_types(code: str) -> dict[frozenset[str], int] | None:
    """``{name pair: bond type}`` from the CCD's own ``chem_comp_bond`` table.

    The dictionary ESMFold2 reads keeps bonds as RDKit sanitizes them, which
    alternates an aromatic ring's single and double bonds differently from the
    CCD file and records a metal-ligand bond as dative. AtomWorks and biotite take
    the file's ``value_order`` and ``pdbx_aromatic_flag``, so a structure that
    AtomWorks reads and one it is given agree on every bond's type.
    """
    from biotite.structure import BondType
    from biotite.structure.info import get_from_ccd

    category = get_from_ccd("chem_comp_bond", code)
    if category is None:
        return None
    orders = {
        "SING": BondType.SINGLE,
        "DOUB": BondType.DOUBLE,
        "TRIP": BondType.TRIPLE,
        "QUAD": BondType.QUADRUPLE,
    }
    aromatic = {
        BondType.SINGLE: BondType.AROMATIC_SINGLE,
        BondType.DOUBLE: BondType.AROMATIC_DOUBLE,
        BondType.TRIPLE: BondType.AROMATIC_TRIPLE,
    }
    types: dict[frozenset[str], int] = {}
    for first, second, order, flag in zip(
        category["atom_id_1"].as_array().tolist(),
        category["atom_id_2"].as_array().tolist(),
        category["value_order"].as_array().tolist(),
        category["pdbx_aromatic_flag"].as_array().tolist(),
        strict=True,
    ):
        kind = orders.get(order, BondType.ANY)
        if flag == "Y":
            kind = aromatic.get(kind, kind)
        types[frozenset((first, second))] = int(kind)
    return types


@cache
def _ccd_template(
    code: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, int], ...]]:
    """``(atom names, (name, name, bond type) bonds)`` of a CCD component.

    The atoms and which pairs are bonded are read the way ESMFold2 reads them: the
    dictionary's molecule with hydrogens removed (``sanitize=False`` keeps the
    chemically significant ones), atoms that carry no name left out. The bond
    types come from the CCD file (:func:`_ccd_bond_types`); a pair it does not
    list keeps the molecule's own type.
    """
    from esm.models.esmfold2.conformers import load_ccd
    from rdkit import Chem

    from esmfold2_atomworks import paths

    ccd = load_ccd(paths.ccd_dir())
    if code not in ccd:
        raise TopologyError(f"CCD component {code!r} is not in the dictionary")
    mol = Chem.RemoveHs(ccd[code], sanitize=False)

    def name_of(atom: Any) -> str:
        return atom.GetProp("name") if atom.HasProp("name") else ""

    names = tuple(name for name in map(name_of, mol.GetAtoms()) if name)
    present = set(names)
    file_types = _ccd_bond_types(code) or {}
    bonds = []
    for bond in mol.GetBonds():
        first, second = name_of(bond.GetBeginAtom()), name_of(bond.GetEndAtom())
        if first in present and second in present:
            kind = file_types.get(frozenset((first, second)))
            bonds.append(
                (first, second, _biotite_bond_type(bond) if kind is None else kind)
            )
    return names, tuple(bonds)


@cache
def _smiles_template(
    smiles: str,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, int], ...], tuple[str, ...]]:
    """``(atom names, (index, index, bond type) bonds, elements)`` of a SMILES ligand.

    The names are ESMFold2's: the element symbol and the canonical rank of the
    atom in the molecule with hydrogens, plus one.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TopologyError(f"RDKit cannot read the SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    for atom, rank in zip(mol.GetAtoms(), AllChem.CanonicalRankAtoms(mol), strict=True):
        atom.SetProp("name", atom.GetSymbol().upper() + str(rank + 1))
    heavy = Chem.RemoveHs(mol)
    names = tuple(atom.GetProp("name") for atom in heavy.GetAtoms())
    elements = tuple(atom.GetSymbol().upper() for atom in heavy.GetAtoms())
    bonds = tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _biotite_bond_type(bond))
        for bond in heavy.GetBonds()
    )
    return names, bonds, elements


def _groups(indices: np.ndarray, res_ids: np.ndarray) -> list[list[int]]:
    """Atom indices of one chain, grouped by residue in order of appearance."""
    _, first = np.unique(res_ids[indices], return_index=True)
    starts = sorted(first)
    bounds = [*starts, len(indices)]
    return [indices[a:b].tolist() for a, b in pairwise(bounds)]


class _Bonds:
    """The bonds found so far, each pair once."""

    def __init__(self) -> None:
        self.found: dict[tuple[int, int], int] = {}

    def add(self, first: int, second: int, kind: int) -> None:
        self.found.setdefault((min(first, second), max(first, second)), kind)


def _within(
    group: list[int],
    code: str,
    names: np.ndarray,
    bonds: _Bonds,
    where: str,
) -> None:
    """The CCD's bonds among the atoms of *group*, which has to be a subset of it."""
    template_names, template_bonds = _ccd_template(code)
    by_name = {names[i]: i for i in group}
    unknown = sorted(set(by_name) - set(template_names))
    if unknown:
        raise TopologyError(
            f"{where}: atoms {unknown} are not atoms of CCD component {code!r}"
        )
    for first, second, kind in template_bonds:
        if first in by_name and second in by_name:
            bonds.add(by_name[first], by_name[second], kind)


def _polymer_residues(entry: Any) -> list[tuple[str, set[str], set[str]]]:
    """``(name, required atoms, allowed atoms)`` of each residue esm makes of *entry*.

    A residue of the standard alphabet has the atoms of esm's own table, no more
    and no fewer. A modified residue, or a letter outside the table, is tokenized
    from its CCD component: every atom that is not flagged leaving is there, and
    the leaving ones may be (a terminal residue keeps them).
    """
    from esm.models.esmfold2.conformers import get_ccd_leaving_atoms
    from esm.models.esmfold2.constants import (
        DNA_1TO3,
        DNA_HEAVY_ATOMS,
        PROTEIN_1TO3,
        PROTEIN_HEAVY_ATOMS,
        RNA_1TO3,
        RNA_HEAVY_ATOMS,
    )

    letters, table = {
        "ProteinInput": (PROTEIN_1TO3, PROTEIN_HEAVY_ATOMS),
        "DNAInput": (DNA_1TO3, DNA_HEAVY_ATOMS),
        "RNAInput": (RNA_1TO3, RNA_HEAVY_ATOMS),
    }[type(entry).__name__]
    codes = [letters.get(letter, "UNK") for letter in entry.sequence]
    modified = set()
    for modification in entry.modifications or []:
        codes[modification.position] = modification.ccd
        modified.add(modification.position)

    residues = []
    for position, code in enumerate(codes):
        if position not in modified and code in table:
            atoms = set(table[code])
            residues.append((code, atoms, atoms))
            continue
        template, _ = _ccd_template(code)
        residues.append(
            (code, set(template) - get_ccd_leaving_atoms(code), set(template))
        )
    return residues


def _ccd_components(
    indices: np.ndarray,
    codes: list[str],
    bonded: bool,
    names: np.ndarray,
    chain: str,
) -> list[list[int]]:
    """Split a ligand chain's atoms into its CCD components, in order.

    ESMFold2 tokenizes the components one after another and, for a chain that
    takes part in a covalent bond, leaves out the atoms the CCD flags as leaving.
    """
    from esm.models.esmfold2.conformers import (
        get_ccd_leaving_atoms,
        get_ligand_ccd_atoms_with_charges,
    )

    groups: list[list[int]] = []
    cursor = 0
    for position, code in enumerate(codes):
        atoms = get_ligand_ccd_atoms_with_charges(code)
        if atoms is None:
            raise TopologyError(f"chain {chain!r}: CCD component {code!r} not found")
        leaving = get_ccd_leaving_atoms(code) if bonded else set()
        expected = [name for name, _element, _charge in atoms if name not in leaving]
        group = indices[cursor : cursor + len(expected)]
        if [names[i] for i in group] != expected:
            raise TopologyError(
                f"chain {chain!r}: component {position} ({code}) should have atoms "
                f"{expected} but the structure has {[names[i] for i in group]}"
            )
        groups.append(group.tolist())
        cursor += len(expected)
    if cursor != len(indices):
        raise TopologyError(
            f"chain {chain!r} has {len(indices)} atoms but its CCD components "
            f"({', '.join(codes)}) account for {cursor}"
        )
    return groups


def build_bond_list(atoms: AtomArray, spi: Any) -> BondList:
    """The bonds of *atoms*, a structure folded from *spi*.

    Args:
        atoms: the structure
            :func:`~esmfold2_atomworks.data.molecular_complex.molecular_complex_to_atom_array`
            returned for the fold, chain ids as in *spi*.
        spi: the ``StructurePredictionInput`` that was folded.

    Raises:
        TopologyError: the structure and the input disagree about a chain, a
            residue or an atom name, so no bond list can be vouched for.
    """
    from biotite.structure import BondList, BondType

    chain_ids = np.asarray(atoms.chain_id).astype(str)
    res_ids = np.asarray(atoms.res_id).astype(int)
    res_names = np.asarray(atoms.res_name).astype(str)
    names = np.asarray(atoms.atom_name).astype(str)

    declared = list(getattr(spi, "covalent_bonds", None) or [])
    bonded_chains = {str(bond.chain_id1) for bond in declared} | {
        str(bond.chain_id2) for bond in declared
    }

    bonds = _Bonds()
    addressable: dict[str, list[list[int]]] = {}
    accounted = np.zeros(len(atoms), dtype=bool)

    for entry in spi.sequences:
        chain = entry.id
        if not isinstance(chain, str):
            raise TopologyError(f"chain id {chain!r} is not a single label")
        indices = np.flatnonzero(chain_ids == chain)
        if indices.size == 0:
            raise TopologyError(f"chain {chain!r} of the input is not in the structure")
        accounted[indices] = True
        kind = type(entry).__name__

        if kind in _POLYMER_INPUTS:
            groups = _groups(indices, res_ids)
            residues = _polymer_residues(entry)
            if len(groups) != len(residues):
                raise TopologyError(
                    f"chain {chain!r} has {len(groups)} residues but its sequence "
                    f"has {len(residues)}"
                )
            for position, (group, (code, required, allowed)) in enumerate(
                zip(groups, residues, strict=True)
            ):
                where = f"chain {chain!r} residue {position + 1} ({code})"
                named = sorted({str(res_names[i]) for i in group})
                if named != [code]:
                    raise TopologyError(f"{where}: the structure names it {named}")
                present = {str(names[i]) for i in group}
                if present - allowed or required - present:
                    raise TopologyError(
                        f"{where}: atoms {sorted(required - present)} are missing "
                        f"and {sorted(present - allowed)} are not its atoms"
                    )
                _within(group, code, names, bonds, where)
            carbon, nitrogen = _POLYMER_INPUTS[kind]
            for before, after in pairwise(groups):
                first = [i for i in before if names[i] == carbon]
                second = [i for i in after if names[i] == nitrogen]
                if first and second:
                    bonds.add(first[0], second[0], int(BondType.SINGLE))
        elif kind == "LigandInput" and entry.ccd is not None:
            groups = _ccd_components(
                indices, list(entry.ccd), chain in bonded_chains, names, chain
            )
            for code, group in zip(entry.ccd, groups, strict=True):
                _within(group, code, names, bonds, f"chain {chain!r} ({code})")
        elif kind == "LigandInput":
            expected, template_bonds, _ = _smiles_template(entry.smiles)
            if [names[i] for i in indices] != list(expected):
                raise TopologyError(
                    f"chain {chain!r}: the SMILES gives atoms {list(expected)} but the "
                    f"structure has {[names[i] for i in indices]}"
                )
            groups = [indices.tolist()]
            for first, second, order in template_bonds:
                bonds.add(int(indices[first]), int(indices[second]), order)
        else:
            raise TopologyError(f"chain {chain!r}: unsupported input {kind}")
        addressable[chain] = groups

    if not accounted.all():
        raise TopologyError(
            f"{int((~accounted).sum())} atoms belong to a chain the input does not name"
        )

    for bond in declared:
        ends = []
        for chain, residue, atom in (
            (str(bond.chain_id1), bond.res_idx1, bond.atom_idx1),
            (str(bond.chain_id2), bond.res_idx2, bond.atom_idx2),
        ):
            try:
                ends.append(addressable[chain][residue][atom])
            except (KeyError, IndexError) as error:
                raise TopologyError(
                    f"declared bond {chain}/{residue}/{atom} is not an atom of the "
                    "structure"
                ) from error
        kind = getattr(bond, "source_bond_type", int(BondType.SINGLE))
        bonds.add(ends[0], ends[1], kind)

    pairs = np.array(
        [
            [first, second, kind]
            for (first, second), kind in sorted(bonds.found.items())
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
    return BondList(len(atoms), pairs)


def _graph(elements: list[str], bonds: list[tuple[int, int, int]]) -> Any:
    """A molecule of bare atoms and bonds of a stated order, for matching.

    RDKit's unspecified bond matches every bond, which is the opposite of what
    matching needs, so a bond whose order is not stated is refused here.
    """
    from biotite.structure import BondType
    from rdkit import Chem

    kinds = {
        int(BondType.SINGLE): Chem.BondType.SINGLE,
        int(BondType.DOUBLE): Chem.BondType.DOUBLE,
        int(BondType.TRIPLE): Chem.BondType.TRIPLE,
        int(BondType.COORDINATION): Chem.BondType.DATIVE,
        int(BondType.AROMATIC): Chem.BondType.AROMATIC,
        int(BondType.AROMATIC_SINGLE): Chem.BondType.AROMATIC,
        int(BondType.AROMATIC_DOUBLE): Chem.BondType.AROMATIC,
        int(BondType.AROMATIC_TRIPLE): Chem.BondType.AROMATIC,
    }
    graph = Chem.RWMol()
    for element in elements:
        graph.AddAtom(Chem.Atom(element.capitalize()))
    for first, second, kind in bonds:
        if kind not in kinds:
            raise LigandIdentityError("a bond of the ligand has no stated order")
        graph.AddBond(first, second, kinds[kind])
    graph = graph.GetMol()
    graph.UpdatePropertyCache(strict=False)
    return graph


def smiles_atom_names(atoms: AtomArray, smiles: str) -> dict[str, str]:
    """ESMFold2's name for each heavy atom of *atoms*, a ligand drawn from *smiles*.

    ESMFold2 names a SMILES ligand's atoms by element and canonical rank, so the
    names a source structure gives them say nothing about which atom is which: a
    bond declared on the source's ``O5`` would land on whatever ESMFold2 calls
    ``O5``. The correspondence is read from the bonds instead. The heavy-atom
    graph of *atoms* has to be the SMILES's, element for element and bond order
    for bond order (aromatic bonds as one kind), and the first such match is the
    mapping. Atoms the match cannot tell apart are equivalent, so which of them
    is taken does not change the molecule. A bond whose order the source does
    not state cannot be matched.

    Returns:
        ``source atom name -> ESMFold2 atom name``.

    Raises:
        LigandIdentityError: the source has no bond list, or its graph is not the
            SMILES's.
    """
    names, template_bonds, template_elements = _smiles_template(smiles)
    element = np.asarray(atoms.element).astype(str)
    heavy = [i for i in range(len(atoms)) if element[i].upper() not in ("H", "D")]
    where = {atom: position for position, atom in enumerate(heavy)}
    if atoms.bonds is None:
        raise LigandIdentityError(
            "the source ligand has no bond list, so its atoms cannot be matched to "
            f"the SMILES {smiles!r}"
        )
    inside = [
        (where[int(a)], where[int(b)], int(kind))
        for a, b, kind in atoms.bonds.as_array()
        if int(a) in where and int(b) in where
    ]
    source = _graph([element[i].upper() for i in heavy], inside)
    target = _graph(list(template_elements), list(template_bonds))
    if source.GetNumAtoms() != target.GetNumAtoms() or len(inside) != len(
        template_bonds
    ):
        raise LigandIdentityError(
            f"the source ligand has {source.GetNumAtoms()} heavy atoms and "
            f"{len(inside)} bonds, the SMILES {smiles!r} {target.GetNumAtoms()} and "
            f"{len(template_bonds)}"
        )
    match = target.GetSubstructMatch(source)
    if not match:
        raise LigandIdentityError(
            f"the heavy-atom graph of the source ligand (elements and bond orders) "
            f"is not that of the SMILES {smiles!r}"
        )
    atom_names = [str(name) for name in np.asarray(atoms.atom_name)[heavy]]
    if len(set(atom_names)) != len(atom_names):
        raise LigandIdentityError("the source ligand repeats an atom name")
    return {atom_names[i]: names[match[i]] for i in range(len(heavy))}


def ccd_name_collisions(atoms: AtomArray, spi: Any = None) -> list[str]:
    """Residue names of hetero residues that a CCD reader would not read back as written.

    A reader that finds a CCD code takes the component from the dictionary, not
    from the file: AtomWorks 2.x rebuilds the residue from the CCD, 3.x refuses
    atoms the component does not have. A name that is not a CCD code is read from
    the file. A hetero residue named for a CCD component is listed when

    * *spi* says its chain is a SMILES ligand, or a CCD ligand of other
      components: whatever its atoms look like, it is not the component named; or
    * its atom names, or the bonds between them, are not a subset of the
      component's. A component with its leaving atoms dropped is a subset and
      reads back as itself.
    """
    from esm.models.esmfold2.conformers import load_ccd

    from esmfold2_atomworks import paths

    hetero = np.flatnonzero(np.asarray(atoms.hetero, dtype=bool))
    if hetero.size == 0:
        return []
    ccd = load_ccd(paths.ccd_dir())
    components: dict[str, set[str]] = {}
    for entry in getattr(spi, "sequences", None) or []:
        if type(entry).__name__ == "LigandInput" and isinstance(entry.id, str):
            components[entry.id] = set(entry.ccd or ())

    names = np.asarray(atoms.atom_name).astype(str)
    residue_of = {
        int(i): (str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.res_name[i]))
        for i in hetero
    }
    members: dict[tuple[str, int, str], set[str]] = {}
    for i, key in residue_of.items():
        members.setdefault(key, set()).add(names[i])
    pairs: dict[tuple[str, int, str], set[frozenset[str]]] = {}
    if atoms.bonds is not None:
        for i, j, _kind in atoms.bonds.as_array():
            key = residue_of.get(int(i))
            if key is not None and key == residue_of.get(int(j)):
                pairs.setdefault(key, set()).add(frozenset((names[i], names[j])))

    collided = set()
    for key, present in members.items():
        chain, _, code = key
        if code not in ccd:
            continue
        if chain in components and code not in components[chain]:
            collided.add(code)
            continue
        template_names, template_bonds = _ccd_template(code)
        known = {frozenset((first, second)) for first, second, _kind in template_bonds}
        if not present <= set(template_names) or not pairs.get(key, set()) <= known:
            collided.add(code)
    return sorted(collided)
