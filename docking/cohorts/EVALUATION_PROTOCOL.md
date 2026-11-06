# Comparison protocol

The receptor input cohorts in this folder are fixed; their membership and receptor
hashes do not depend on docking, features or model results.

For method comparisons, the common test cohort is the intersection of available pairs
across all eighteen docking inputs: Vina, smina and GNINA, both docking scenarios, and
three runs. A pair must have at least one pose with a finite native score and finite,
nonnegative MBG and whole-ligand RMSD labels in every input. Accurate and inaccurate
poses are equally eligible. Model predictions, success thresholds and prediction
cutoffs do not select this cohort. Its IDs and hashes are recorded before test
performance is computed. `model.cohorts` defines and tests this protocol.

The same test pairs are used for all six methods. External baseline outputs are
matched using the holo system and ligand IDs; their apo suffix is removed for this
lookup. Missing outputs, missing labels and negative RMSD sentinels count as failures
at the stated denominator. AutoDock-GPU, GPDOCK and MetalloDock are also reported on
all 156 curated pairs, next to the per-program evaluable sets.

The GNINA leakage strata (Fig. S12) use PDB-level and cluster-level overlap with
GNINA's training set; within each stratum, all compared methods use the same pairs.
The GPDOCK and MetalloDock unseen-system populations (Figs. S13, S14) mostly lie
outside the curated 156-pair test set; each comparison uses the intersection of
labelled pairs across its three methods and all runs. Each figure records the cohort
it uses.

For the affinity panels (Figs. S10, S11), a missing score cannot be represented as a
failed correlation point. Each scenario therefore uses the pairs with an affinity
annotation and a finite top-1 score for Vina, smina, GNINA, AutoDock-GPU and
MetalloDock in all three runs, matched per ligand. The benchmark pickle records this
subset under `comparison_cohorts.affinity_<scenario>`. It is selected by
availability, without success thresholds or model predictions; every method in a
panel uses exactly the same pairs.
