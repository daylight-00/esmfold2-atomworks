Structures used by the parity tests, so that they run without an AtomWorks checkout.

| file | entry | note |
|---|---|---|
| `6lyz.bcif` | PDB 6LYZ | lysozyme, binary CIF |
| `2hhb.cif.gz` | PDB 2HHB | haemoglobin |
| `4q8n.cif.gz` | PDB 4Q8N | protein with a zinc ion |
| `1a8o.cif` | PDB 1A8O | four selenomethionines (`MSE`) |

The files are PDB entries as distributed by AtomWorks' test data. The deposited
`1A8O` atom `ASP A 2 CG` is labelled `CG`, as deposited; AtomWorks' own test copy
renames it `XYZ` to exercise an unknown heavy atom, which AtomWorks 3.0 rejects.
