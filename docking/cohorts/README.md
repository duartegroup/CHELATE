# Receptor input cohorts

These CSVs list, for each docking program and split, the system/ligand pairs whose
prepared receptors are used for docking, with the source and receptor hashes, the
docking box and the reason for every exclusion. They were fixed from structure and
preparation provenance before docking, labelling or evaluation, and are not changed in
response to docking, feature or model results. No binding-affinity values are included.

A pair is excluded if its receptor contains atoms of the target ligand, or if a source
heavy atom is missing within 8 Å of the closed docking box (Vina's interaction
cutoff). The box is the ligand extent plus 4 Å on every side
(`docking_utils.get_box`); coordinates are compared with a 0.01 Å tolerance. Missing
atoms further from the box are recorded (`missing_atoms_outside_binding_site`) but do
not exclude the pair. These are isolated structures; periodic images do not apply.
The same receptor is used for all three docking programs, and every selected PDBQT
input passes format validation and the binding-site audit.

| Split/scenario | Curated pairs | Excluded | Retained |
|---|---:|---:|---:|
| Training (up to five per cluster) | 4,083 | 472 | 3,611 |
| Validation | 1,073 | 120 | 953 |
| Test, re-docking | 156 | 14 | 142 |
| Test, cross-docking | 156 | 23 | 133 |

Paths (`original_source`, `ligand_sdf`) are relative to `SYSTEMS_ROOT`; the prepared
receptor is identified by `receptor_sha256`. `whole_protein_defect` records whether the
selected receptor misses any source atom anywhere in the protein (diagnostic only).
GNINA receives the same explicit centre and size as Vina and smina and uses its
default `autobox_extend` expansion where needed for input-conformer rotation.
The receptor exclusion rule uses the shared explicit box for every engine.
`summary.json` gives the counts per program and split.

The table is identical for all docking programs. Test CSVs contain both scenarios;
training and validation use re-docking only. Model training and evaluation also
require poses and RMSD labels, so the numbers of labelled pairs are lower (reported
with the results).

## Common test cohort

`comparison_common_test.csv` contains the 118 test pairs evaluable for all three
docking programs, both docking scenarios and all three runs (see
`EVALUATION_PROTOCOL.md`). Its companion JSON records the eighteen input hashes, the
code hash and the per-input eligible counts; no models, prediction cutoffs or success
thresholds were used. The per-program evaluable sets are Vina 137/130, smina 135/128
and GNINA 141/132 pairs (re-docking/cross-docking). The 118-pair list is the common
population for all method comparisons, external baselines included; the per-program
and all-156 tables are reported alongside.
