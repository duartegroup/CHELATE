# Figures

Notebooks and scripts that make the figures of the CHELATE paper and its SI, with the
inputs they read. Each notebook runs from its own directory and writes its PDFs next to
it (`plots/` or `figures/`, not tracked by git).

Main test comparisons use the **118-pair common cohort**: the test pairs evaluable for all three docking programs, both scenarios and all three runs, fixed from input and label availability before the models were evaluated on the test set. Missing external outputs count as failures. Affinity and unseen-system panels use their explicitly matched subsets, recorded in the benchmark pickle's metadata. Separate per-engine and fixed-/156 tables are in `model/NUMBERS.md` section 7; these denominators must not be mixed.

| Figure | Made by | Inputs |
|---|---|---|
| Fig. 2, S6–S14, S26 | `benchmark/benchmark.ipynb` | `benchmark/benchmark_results.pkl` |
| Fig. 3, Fig. 4 (selective filter), S16–S24, S28, all-systems SI figure | `model/model.ipynb` | `model/csv_files/`, `model/test_system_metals.json`, `../model/data/EC_numbers.csv` |
| Fig. S1–S5, S15 | `dataset/dataset_statistics.ipynb` | `inputs/`, `dataset/*.pkl`, curated metal JSON and distance cache (both untracked; from the Zenodo record, see below) |
| Fig. S25, S27 | `metal_identity/make_metal_identity_figures.py` | `metal_identity/probe_*/` |
| Fig. 5 (case studies) | `case_studies/export_case_studies.py`, then PyMOL | `case_studies/current_cases.json`, cross-docked poses and receptors |

`model/data/EC_numbers.csv` (Fig. S22): EC numbers were retrieved from the RCSB PDB
(`rcsb_polymer_entity.rcsb_ec_lineage`) for each test-set entry on 29 April 2026.

`model/NUMBERS.md` lists every model number quoted in the paper;
`model/scripts/paper_numbers.py` regenerates it from `model/csv_files/`.

## Rebuilding the inputs

The inputs are derived from the docking results, which are too large for git (they are
in the Zenodo bundle). On a machine that has them, run each command from the repository
root into a scratch directory, check the output, then copy the named files into `figures/`.

| Input | Command |
|---|---|
| Common comparison feature inputs | `python -m model.stage_comparison_features --features-dir docking/features --cohort docking/cohorts/comparison_common_test.csv --output-dir <scratch>/comparison_features` (scratch inputs for the rows below; keeps the original numeric CSV text) |
| `model/csv_files/` | `python -m model.test_eval --features_dir <scratch>/comparison_features --cohort docking/cohorts/comparison_common_test.csv --output_dir <scratch>/evaluation` (retained-system mode, no `--keep_best_pose`); copy the result CSV/JSON files into `figures/model/csv_files/` |
| `model/csv_files/distance_baseline_{test_curve,trainval_curve,summary}.csv` | `python -m model.distance_baseline --features_dir <scratch>/comparison_features --output_dir <scratch>/distance --chelate_grouped_csv <scratch>/evaluation/plot_repro_grouped_metrics.csv`; copy the three CSVs with the `distance_baseline_` prefix |
| `model/csv_files/mbg_sr_vs_training_fraction_predCutoff=VINA-0p8_SMINA-0p85_GNINA-0p8.csv` | `python figures/model/scripts/run_sweep.py <scratch>/sweep --pred-cutoffs --features-dir <scratch>/comparison_features` |
| `model/csv_files/cutoff_selection_validation.csv`, `model/params/pred_cutoffs.json` | `python -m model.select_cutoff --runtime-dir <new scratch dir>` (train-only fits and validation only; freeze and commit these choices before any model test evaluation) |
| `benchmark/benchmark_results.pkl` | `python figures/benchmark/build_benchmark_results.py --cohort docking/cohorts/comparison_common_test.csv --unseen_dir figures/benchmark/unseen_systems --output <scratch>/benchmark_results.pkl` |
| `case_studies/current_cases.json` exports | `python figures/case_studies/export_case_studies.py --systems-root <case_inputs> --output-dir <new outside-git folder>`; the Zenodo record supplies the four systems in `case_inputs/` |
| `dataset/spearman_cache.pkl` | `python figures/dataset/build_spearman_cache.py --features_dir <scratch>/comparison_features --output <scratch>/spearman_cache.pkl` |
| `metal_identity/probe_npose{1,100}/` | `python -m model.predict_metal_identity --model model/params/VINA_rmsd=1.0_best_model.cbm --systems-dir <probe_inputs> --n-poses 100 --output-dir <scratch>/probe_npose100`; repeat with `1`. The bundle supplies the frozen redocking receptor inputs in `probe_inputs/` |
| `dataset/chelation_cache.pkl`, `dataset/metal_ligand_distances_cache.pkl` | `python figures/dataset/build_dataset_caches.py --chelation-output <scratch>/chelation_cache.pkl --distances-output <scratch>/metal_ligand_distances_cache.pkl`; the distance cache is not tracked |

For the per-engine and /156 accounting, also run `python -m model.test_eval --output_dir <scratch>/engine_evaluable` on the full feature directory, then `python -m model.cohort_report --common-evaluation <scratch>/evaluation --eligible-evaluation <scratch>/engine_evaluable --common-cohort docking/cohorts/comparison_common_test.csv --curated-cohort docking/cohorts/VINA_test.csv --output <scratch>/cohort_report`. Copy its three CSVs into `figures/model/csv_files/` before regenerating `NUMBERS.md`.

`benchmark/compare_benchmark_pickles.py` compares two pickles leaf by leaf; use it to
check that a rebuilt pickle matches the committed one.

## `inputs/`

Files the repository pipeline does not produce but the figures need:

- `final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit_NEW.csv`:
  the curated dataset table with splits (Fig. S2; cluster leakage sets of Fig. S12).
- `final_dataset_..._resplit_BINDING_PRIO.csv`: dataset IDs and split metadata for
  the binding-prioritised test set (Figs. S10, S11). It is committed without its
  affinity column; `benchmark/build_benchmark_results.py` joins PLINDER's
  `ligand_binding_affinity` annotation back from the `_NEW` table (same rows).
- `crossdocked_v1.3_train_unique.csv`: GNINA's training set (Fig. S12).
- `split_clusters.csv`: PLINDER cluster assignments (Fig. S12).

Two inputs of `dataset_statistics.ipynb` are too large for git: the curated metal JSON
(`data_curation/cached_data/metal_ligand_data_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.json`,
118 MB) and the distance cache `dataset/metal_ligand_distances_cache.pkl`. Both are in the
Zenodo record under `working_package/` at the same paths; copy them there before running
the notebook from a clean checkout.

## Environment

The `chelate` conda environment plus `matplotlib`, `seaborn` 0.13.2, `scipy`, `pyarrow` (for
the distance cache) and `jupyter`.

The prediction cutoffs are Vina 0.80 Å, smina 0.85 Å and GNINA 0.80 Å, chosen on the validation set before the test evaluation (`model/params/pred_cutoffs.json`). Case-study reference MBGs use the holo metal coordinates, matching the benchmark labels.

The distance cache has 772,790 rows and the chelation cache 40,044 entries (see `dataset/cache_provenance.json`); `dataset/build_dataset_caches.py` rebuilds both. Neither cache applies periodic boundary conditions to the assembled PLINDER structures.

The metal-identity probe ranks the ten entries in `chelate.constants.EXPLICIT_METALS`:
Mn, Co, K, Cu, Mg, Zn, Fe, Ni, Ca and Na. Uniform top-1 chance is therefore 10%.
`ALL_ZERO` is a separate diagnostic control and is excluded from the primary metal
ranking. The number of metal types observed among eligible true labels does not
change that candidate-space chance level. Each probe summary records this protocol.
