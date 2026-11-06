# Receptor preparation, repair and integrity checks

Receptors are prepared from the sanitized system CIF with Meeko. Most receptors
are prepared directly from the CIF. Receptors that Meeko cannot prepare directly
go through a fixed sequence of general repair retries. Every receptor, however it
was prepared, must pass the same source-atom and metal audits before it is docked.
Repairs are general rules and contain no system-specific exceptions.

The code is split as follows:

- `run_docking.py` prepares receptors directly and calls
  `receptor_gate.require_complete_receptor` before any docking engine runs.
- `prepare_pairs.prepare_original_pair` prepares one isolated system-ligand pair
  and tries the repair policies in order until one passes all audits.
- `repair_receptors.py` is the batch command-line tool for receptors that failed
  direct preparation. It applies the same repairs and audits.
- `receptor_integrity.py` compares a prepared receptor with its pair-specific
  source structure.
- `receptor_pdb_cache.py` converts prepared PDBQT receptors to PDB for feature
  generation and the metal-identity probe.

## Preparation policies

The direct policy runs Meeko on the sanitized CIF. If it fails, PDBFixer replaces
nonstandard polymer residues (for example MSE with MET, without moving
coordinates), removes heterogens and adds missing heavy atoms. Valid
single-character chain IDs are preserved. Separately prepared unsupported atoms,
including metal ions that Meeko does not support, are then merged back.

Deuterium element fields are normalized to hydrogen before Meeko reads the PDB.
This changes the isotope label without moving the atom or changing its identity.

### Hydrogen and terminal-oxygen retry

With `--repair-input-geometry`, a failed preparation gets one additional retry:

- Supplied hydrogen atoms are removed. Meeko assigns hydrogens using its standard
  residue templates, resolving invalid input hydrogen connectivity and template
  ambiguities without selecting templates for individual systems.
- An OXT absent from the source CIF is treated as an added terminal oxygen. Its
  position is reflected from the observed carbonyl oxygen across the CA-C axis.
  This preserves the observed C-O bond length and peptide plane. Source heavy
  atoms, including an OXT already present in the source, are unchanged.

The retry addresses PDBFixer-added OXT clashes and hydrogen connectivity
failures. It does not suppress remaining valence errors. Failed receptors remain
listed as preparation failures.

### Residue-template retry

With `--repair-residue-templates PATH`, one further retry fits the standard
PDBFixer residue templates to the observed heavy atoms and rebuilds only atoms
that PDBFixer added. The fit requires at least three non-collinear observed
anchors and uses a proper rigid rotation without reflection. Source heavy-atom
coordinates stay fixed, including atoms renamed during residue replacement
(such as MSE SE renamed to MET SD). Added OXT atoms keep the terminal-oxygen
repair above. The attempt records each moved atom, its anchors, and template
checksums. Meeko must then prepare a nonempty receptor, and the same metal and
chain audit must pass. This is a general retry for invalid added-atom geometry;
it contains no per-system template selections.

### Free amino-acid residue templates

`prepare_pairs.prepare_original_pair` tries combined terminal templates only after
all preceding direct, PDBFixer, geometry and residue-template retries fail. Meeko's
separate N-terminal and C-terminal templates can omit an isolated amino acid that
has both termini and no peptide neighbours. A successful Meeko exit alone does
not detect this loss.

`free_residue_templates.py` derives the additional templates from the exact Meeko
library installed in the preparation environment. It combines the library's
N-terminal amino group and C-terminal carboxylate, retaining side-chain states
and the library's template preference order. Templates with additional covalent
links are not combined. No existing template is replaced. The overlay is used in
an isolated Meeko subprocess; installed Meeko files stay unchanged.

The retry reads existing prepared PDBs without moving heavy atoms. Each attempt
records the parent PDB, base template library, generated overlay and backend
checksums. It must pass the same complete source-atom audit, target-ligand
exclusion, chain mapping and metal-retention checks as every other preparation
route. A retained residue is not sufficient if any other required atom is absent.

## Metal retention

PDBFixer's `removeHeterogens()` also removes metal ions that Meeko supports, such
as Mg, Zn, Mn and Ca. Restoring only the separately prepared unsupported-ion file
would lose those supported ions. The merge therefore also reads monatomic metal
residues from the sanitized CIF and restores missing sites at their source
coordinates and chain IDs. Existing unsupported-ion records keep their charges.
Added supported ions use the Meeko residue-template charge, or an explicit
source-CIF formal charge if no matching template exists. Missing charge
information is an error; the code does not guess an oxidation state.

The audit checks every source metal-ion site, as well as the separately prepared
unsupported records, against the merged receptor. It checks chain IDs, record
multiplicity and coordinates at PDBQT precision. These assembled structures use
direct Cartesian coordinates, without periodic images.

## Source-atom audit

`receptor_integrity.audit_original_receptor` compares the heavy atoms of a
prepared receptor with the pair-specific source structure, not with shared
intermediates. Meeko's successful exit with `--allow_bad_res` is not sufficient,
because that option can omit source protein residues. Hydrogens, waters,
additional models and atoms added during preparation are not missing-source
defects. Alternate sites of the same source atom require one retained conformer.
Reduce's amide flips exchange N/O identities, so polymer positions are compared
geometrically; metals must also match by element.

With a docking box, only missing atoms at most 8 A from that box exclude a pair.
Target-ligand atoms in the receptor always exclude it. All missing atoms and the
whole-protein diagnostic are kept in the result. `run_docking.py` applies this
check through `receptor_gate.require_complete_receptor` to every receptor,
including cached ones, before docking.

### Docking box

GNINA 1.3.3 has `autobox_extend=true` by default. Its source can enlarge each
axis to fit the maximum ligand span, so `--autobox_add 4` alone does not guarantee
the same effective box as `docking_utils.get_box`. The smina binary has no
corresponding automatic-extension option.

`box_options.fixed_box_options` therefore passes the centre and sizes explicitly
to Vina, smina and GNINA and disables GNINA's automatic extension. The receptor
gate audits that same box.

Source: [GNINA 1.3.3 main.cpp](https://github.com/gnina/gnina/blob/v1.3.3/gninasrc/main/main.cpp),
option definition and `autobox_extend` branch; [maximum-span implementation](https://github.com/gnina/gnina/blob/v1.3.3/gninasrc/lib/model.cpp).

## Batch repair with `repair_receptors.py`

By default the tool collects the pairs whose GNINA redocking error file reports a
Meeko receptor failure and whose merged receptor is missing or invalid. An
explicit `--tasks-json` manifest instead lists `split`, `system_id` and
`ligand_id` for each pair.

- `--retry-direct` repeats direct Meeko preparation from the current source CIF,
  writes and validates a fresh candidate, and does not invoke PDBFixer. It keeps
  the same Meeko flags and applies the same merge and source-metal audit.
- `--retry-normalized` reuses the PDBFixer PDB saved by a failed attempt instead
  of running a new missing-atom minimization.
- `--retain-metals-only` keeps an already prepared protein and restores missing
  source ions.
- `--metal-template-json` sets the Meeko residue-template file used for metal
  charges and records its checksum.
- `--stage-only` writes audited repairs to the work directory without replacing
  existing receptors. The audit record lists the files that would be installed.

Each attempt keeps its full Meeko log and repair and audit metadata. Existing
outputs are replaced only after a candidate passes validation and auditing, and
the replaced files are kept with a timestamp suffix. Poses docked against a
replaced receptor must be docked again before features are generated. The result
JSON and failure list distinguish successful repairs from unresolved failures.
`--allow-failures` lets a batch finish with that explicit failure list; it does
not classify failed receptors as successfully prepared.

## Cached preparations

Cached preparation records are rechecked with the current audits and matched to
the exact source structure and target ligand before reuse. A defective cache, or
one whose source or receptor has changed, is rejected without changing it; retry
it in a new isolated directory.

## Converted receptor PDBs

Feature generation and the metal-identity probe use the same receptor conversion
helper. A cached PDB is reused only when its provenance matches the exact PDBQT
and both file hashes, and all atom serials, identities, elements and coordinates
match. Otherwise the PDB is rebuilt with OpenBabel and the source atom identities
are restored. Existing PDBs and provenance files are archived under
`.runtime/receptor_pdb_cache/` before replacement. A failed conversion cannot fall
back to the existing PDB. The probe uses the ligand-specific receptor rather than
a generic `receptor_merged.pdb` that may belong to another removed ligand.

The conversion identity check compares coordinates as exact finite numbers.
OpenBabel may serialize `-0.000` as `0.000`; these coordinates are identical and
are accepted. A movement of even 0.001 Angstrom is rejected. The converted
coordinate text and bonds are kept when the source atom identity is restored.

## Explicit receptor connectivity

Receptor PDBs converted from PDBQT contain an explicit OpenBabel CONECT network.
RDKit's default proximity bonding can add extra bonds to that complete network.
In the 3BY0 apo receptor, for example, it would add an arginine N--C bond at
1.903 A and a terminal O--O bond at 0.591 A, giving N valence 5 and O valence 3.
The explicit network sanitises without those bonds.

`chelate.receptor.load_receptor_pdb` therefore keeps complete explicit
connectivity, in both benchmark feature generation and public inference. It
preserves atom records, coordinates, metals and every explicit bond. PDBs with
missing or partial connectivity use normal RDKit bond perception. A matching
conversion-provenance hash marks a PDBQT conversion as an explicit-connectivity
input even when a few atoms are isolated. If sanitisation still fails, bond orders
inside the failing standard amino-acid residue are restored from RDKit's named
residue templates. Bond endpoints, cross-residue links, formal charges, atoms and
coordinates are preserved. Unknown residues must still pass sanitisation. The
rule applies to all receptors; there are no system-specific exceptions.

## Ligand charges with coordinated metals

Meeko's ligand charge calculation removes coordinated metals from an auxiliary
copy of the ligand and caps the freed donor atoms. A dative donor contributes zero
valence to the metal bond, so removing that bond does not justify adding a
hydrogen to the donor, and doing so can raise a valence error.
`python -m docking.prepare_ligand` runs Meeko unchanged first. Only after a
valence exception with such a donor does it repeat the auxiliary charge
calculation without that hydrogen cap. Covalent metal bonds keep Meeko's cap and
charge reassignment. The ligand's atoms, bonds, coordinates and formal charges are
unchanged. Unsupported native atom types are reported as engine failures and are
never mapped to another element. No installed environment is patched.
