<p align="center">
  <img src="assets/logo.png" alt="CHELATE logo" width="260">
</p>

# CHELATE: High precision structure prediction for metalloproteins

[![CI](https://github.com/duartegroup/CHELATE/actions/workflows/ci.yml/badge.svg)](https://github.com/duartegroup/CHELATE/actions/workflows/ci.yml)

## Overview

This project focuses on evaluating and improving the performance of molecular docking tools for metalloproteins. This is done in three stages:

1. Curating a comprehensive metalloprotein-ligand dataset by subsetting [PLINDER](https://www.plinder.sh),
2. Benchmarking six docking programs (AutoDock Vina, smina, GNINA, AutoDock GPU, GPDOCK, and MetalloDock) in both re-docking and cross-docking, and
3. Developing CHELATE, a machine-learning post-docking filter that predicts how far a pose places the metal-binding group from its crystallographic position and discards the poses it judges unreliable.

## Citation

If you use CHELATE, please cite the accompanying paper (reference added on publication).

## Repository Structure

[chelate](chelate/) is the installable package containing the core CHELATE functionality — everything needed to re-rank docked poses. [data_curation](data_curation/) contains the code for curating the metalloprotein-ligand dataset, [docking](docking/) the code for running traditional docking programs, [model](model/) the training and evaluation scripts for the re-ranking models, and [figures](figures/) the notebooks and scripts that produce every figure in the paper and its SI, each documenting the inputs it reads.

## Using CHELATE

To apply CHELATE to your own docked poses, install the `chelate` package and its required Python dependencies from the repository root:

```bash
pip install .
```

This installs CHELATE and its trained models. Running CHELATE on existing poses does not require the full benchmark setup. To run the data curation, docking, training, or evaluation scripts, follow [Full benchmark setup](#full-benchmark-setup).

Trained models for Vina, Smina and GNINA ship with the package, so re-ranking a set of poses is a single call:

```python
from chelate import predict_poses

result = predict_poses(
    receptor="receptor_prepared.pdb",  # prepared receptor, including the metal
    poses="vina_out.pdbqt",  # docking output
    method="VINA",
)

result.metals  # metals detected in the receptor, e.g. ['Zn']
result.reranked()  # poses passing the filter, in docking-score order
result.top1()  # CHELATE's pick, or None if no pose is trusted
result.baseline_top1()  # what the docking program ranked first
```

No crystal structure is required — CHELATE predicts how far each pose's metal-binding group is likely to sit from the true one and uses that prediction to *filter* the pose list; the docking program's own score (`docking_score` for Vina and Smina, `CNNscore` for GNINA) then ranks the survivors. The default cutoff depends on the docking program: 0.80 Å of predicted MBG RMSD for Vina, 0.85 Å for Smina and 0.80 Å for GNINA (`chelate.DEFAULT_CUTOFFS`), the operating points used in the paper, chosen on the validation set as the cutoffs with the highest top-1 MBG success among retained systems while retaining at least half of them. Pass `cutoff=` to trade coverage for precision.

Supported inputs:

- **Poses**: Vina and Smina PDBQT output (a Vina SDF exported with Meeko's `mk_export.py` also works), GNINA SDF or `.sdf.gz`. The ligand must have been prepared with Meeko, which writes the `REMARK SMILES` records needed to rebuild it from PDBQT.
- **Receptor**: a PDB file including the metal(s). The bundled models were trained on Meeko-prepared receptor PDBQT files converted to PDB with Open Babel (`obabel receptor.pdbqt -O receptor.pdb`); preparing your receptor the same way keeps the charge features consistent with training.

Optional extra: `chelate[eval]` to score predictions against a known crystal pose (`add_reference_mbg_rmsd` for the metal-binding-group RMSD that CHELATE predicts, `add_reference_rmsd` for the whole ligand). See [model/example/chelate.ipynb](model/example/chelate.ipynb) for a worked example.

To run the tests: `pip install -e ".[test]"` and `pytest`. CI runs them, together with `ruff check` and `ruff format --check`, on every push and pull request.

## Full benchmark setup

The benchmark scripts in `data_curation/`, `docking/`, and `model/` run directly from this repository checkout. Run the following setup commands from the repository root. Each workflow example below also starts there before changing to its script directory.

We use two separate conda environments for data curation and for docking, feature generation, training, and evaluation due to some version incompatibilities. Create them with:

```bash
conda env create -f plinder_env.yml
conda env create -f docking_env.yml
```

Additionally, [Scrubber](https://github.com/forlilab/molscrub) needs to be installed from source:
``` bash
conda activate docking_env
git clone https://github.com/forlilab/molscrub.git
cd molscrub
python -m pip install -e .
cd ..
```

Reduce2 requires [geostd](https://github.com/phenix-project/geostd/), which needs to be downloaded and placed in the `data_curation` directory.
```bash
git clone https://github.com/phenix-project/geostd.git
mv geostd data_curation
```

Finally, you need [AutoDock Vina ](https://github.com/ccsb-scripps/AutoDock-Vina/releases), [SMINA](https://sourceforge.net/projects/smina/), and GNINA (can be installed in various ways, we recommend using a [Docker image](https://hub.docker.com/r/gnina/gnina)).

To reproduce the benchmark results, you additionally need [AutoDock](https://github.com/ccsb-scripps/autodock-gpu) (we used version `AutoDock-GPU/1.5.3-GCC-11.3.0-CUDA-11.7.0`), [GPDOCK](https://github.com/wangkai-zhku/GPDOCK) (clone the repo and move the executables), and [MetalloDock](https://github.com/SII-ZhangHui/MetalloDock) (clone the repo and follow their instructions).

The scripts in `docking/` and `model/` import the shared feature schema from the `chelate` package. Install CHELATE into `docking_env`, then install Optuna for hyperparameter optimisation, scikit-learn for evaluation metrics, and tqdm for docking progress reporting:

```bash
conda activate docking_env
python -m pip install -e .
python -m pip install optuna scikit-learn tqdm
```

Activate `plinder_env` for data curation and `docking_env` for the docking and model scripts.

### Working directories and `SYSTEMS_ROOT`

Default data paths are resolved from the repository, so these scripts can be run from
any working directory. Explicit relative paths on the command line are still relative
to the working directory.

| Scripts | Run from | Repository-based defaults |
|---|---|---|
| `data_curation/export_crossdocking_inputs.py` | anywhere | `data_curation/cached_data/`, `docking/` |
| `docking/run_docking.py` | anywhere | `data_curation/cached_data/`, `docking/` |
| `docking/generate_features.py` | anywhere | `data_curation/cached_data/`, `docking/`, `docking/features/` |
| `model/*.py` (run as `python -m model.<script>`) | repository root for module execution | `model/`, `docking/` |

The docked systems live in one directory that holds the `train_systems`, `val_systems` and `test_systems` folders. Set the `SYSTEMS_ROOT` environment variable to its **absolute** path once, so that every script uses the same folders:

```bash
export SYSTEMS_ROOT=$PWD/docking   # run from the repository root
```

`SYSTEMS_ROOT` is the default for `run_docking.py --systems_root`, `generate_features.py --systems-root`, `data_curation/export_crossdocking_inputs.py --systems-root`, and the cross-docked receptor lookup in `model/model_utils.py`. If it is unset, all four use `<repository>/docking`. The examples below assume that default.

Update the executable and PLINDER storage paths in [docking_params.py](docking/docking_params.py), and the curated dataset and feature paths in [model_params.py](model/model_params.py), for your local setup. Use absolute executable paths when the same setting is used from both `docking/` and `model/`. Set `MMTBX_CCP4_MONOMER_LIB` in `docking_params.py` to the absolute path of the downloaded `data_curation/geostd` directory so Reduce2 can find it from any working directory.

## Dataset

The full PLINDER dataset is hundreds of GB large, and all curation steps can take a long time. The precurated dataset, all docking poses, the trained models and the figure inputs are available from Zenodo (DOI: added on publication). The archive contains this repository as `working_package/`, with the curated files under `working_package/data_curation/cached_data/` and the docked systems under `working_package/docking/`.


Alternatively, you can generate it yourself by downloading the PLINDER dataset and running [generate_dataset.py](data_curation/generate_dataset.py)
```bash
cd data_curation
conda activate plinder_env
plinder_download --release 2024-06
python generate_dataset.py --run_resplit --ligand_metal_min_threshold 4.0 \
                           --good_apo_threshold 70.0 \
                           --save_superimposed_structures \
                           --max_ligand_mw 750.0 \
                           --plot_split
```

To explicitly disable a boolean option, use the `--no-<option>` flag, e.g.:
```bash
python generate_dataset.py --no-plot_split --no-save_superimposed_structures
```

After downloading our precurated data or generating the dataset yourself, you should have the following the files, where `{identifier}` is a placeholder for a string that identifies the generated dataset files based on the parameters chosen when running `generate_dataset.py`.
- `data_curation/cached_data/final_dataset_{identifier}.csv`: Final curated dataset metadata.
- `data_curation/cached_data/metal_ligand_data_{identifier}.json`: Metal positions and ligand-metal distances.
- `data_curation/cached_data/alt_structures_final.csv`: Alternative structures and similarity metrics.
- `data_curation/cached_data/matched_final.csv`: Matched apo structures.
- `data_curation/cached_data/split_clusters.csv`: Cluster of each system after resplitting.
- `data_curation/cached_data/train_cluster_representatives.csv`: Single representative of each train cluster.
- `data_curation/cached_data/cluster_representatives_5.csv`: Up to five randomly chosen systems from each train cluster. This is the training subset that is docked and featurised (see below).
- `data_curation/split_plots/`: Dataset statistics visualisations.

The default identifier is `distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit`, corresponding to default parameters.

#### Training subset

Only up to five randomly chosen systems per train cluster are docked. [sample_train_representatives.py](data_curation/sample_train_representatives.py) writes this subset from `split_clusters.csv`:

```bash
cd data_curation
python sample_train_representatives.py \
    --dataset-csv cached_data/final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.csv \
    --output cached_data/cluster_representatives_5.csv
```

The precurated `cluster_representatives_5.csv` (3,755 systems from 1,507 clusters) is the file used for the published results. Its cluster sizes match this script exactly (min(5, cluster size) systems per cluster), but the random choices within larger clusters cannot be regenerated, so keep the shipped file to reproduce the paper. The script refuses to overwrite an existing output unless `--overwrite` is given. `run_docking.py` and `generate_features.py` both default to this file for the `train` split.

#### Cross-docking inputs

Cross-docking uses the apo structure superimposed onto each holo system. `run_docking.py` expects it as a Biotite mmCIF file at `{SYSTEMS_ROOT}/{split}_systems/{system_id}/{system_id}__{target_system}.cif`. `generate_dataset.py --save_superimposed_structures` saves the aligned structures as PDB files under `data_curation/superimposed_structures/` instead (with chain IDs shortened to one character), so write the docking inputs with [export_crossdocking_inputs.py](data_curation/export_crossdocking_inputs.py), which re-runs the alignment for one split and writes the CIF files in that layout:

```bash
conda activate plinder_env
python data_curation/export_crossdocking_inputs.py --split test
```

Existing CIF files are skipped unless `--overwrite` is given. The published test-set CIFs come from the alignment in `superimpose_structures.py`; on eight checked test systems they match this script's output in atoms, coordinates, chains and metals, and differ only in CIF metadata, listed in the script's docstring.

## Docking Benchmark
Update the paths in [docking_params.py](docking/docking_params.py) according to your local setup, then run `run_docking.py` from the repository root (with `SYSTEMS_ROOT` set as described above, or pass `--systems_root`):

```bash
conda activate docking_env
python docking/run_docking.py --split test \
                      --method VINA \
                      --id 42 \
                      --redocking \
                      --add_hydrogen \
                      --exhaustiveness 32 \
                      --num_modes 100 \
                      --energy_range 20.0 \
                      --box_extension 4.0 \
                      --dataset_path data_curation/cached_data/final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.csv
```

This will run re-docking on the test set using Vina; omit `--redocking` (or pass `--no-redocking`) for cross-docking, which is the default. See [run_docking.py](docking/run_docking.py) for further details about the arguments. The script will iterate over the provided dataset (`dataset_path`), which expects the output format of [generate_dataset.py](data_curation/generate_dataset.py), and will attempt to dock every system matching the specified split; for `train`, only the systems in `--representative_train_systems_path` (default `data_curation/cached_data/cluster_representatives_5.csv`) are docked. For each split and PLINDER system, a separate folder `{SYSTEMS_ROOT}/{split}_systems/{system_id}` will be created, in which the receptor, ligand, prepared structures, and docked systems are saved. Systems are copied from the PLINDER local storage (set in [docking_params.py](docking/docking_params.py)). Cross-docking additionally needs the superimposed apo CIF files described under [Cross-docking inputs](#cross-docking-inputs).

GNINA uses the Scrubber-prepared `_scrubbed.sdf` ligand at pH 7.4, the same explicit box centre and size as Vina and smina (`--center_x/y/z`, `--size_x/y/z`), which GNINA enlarges where needed so that the input conformer can rotate (`autobox_extend`, the default), and a deterministic seed equal to `--id`. Set `GNINA_PATH` to a GNINA executable or command if the default container command is unavailable. The tested GNINA v1.3.3 binary does not accept `--energy_range`; this pipeline passes that option only to Vina and Smina. GNINA pose and result filenames include `prepPH=7.4`, the pH at which the ligand was prepared. Vina and smina draw their own random seed for each run; no seed is passed to either engine.

If a train or validation GNINA run reports Meeko receptor-preparation failures,
run [repair_receptors.py](docking/repair_receptors.py) after `run_docking.py`, then
repeat the failed docking jobs. The fallback uses PDBFixer to replace nonstandard
polymer residues and add missing atoms before calling Meeko, while preserving and
merging the separately prepared metal ions. Its defaults locate the data and run state
relative to the repository; use `--data-root`, `--shards-dir`, `--error-filename`,
`--python`, and `--meeko` for another layout:

```bash
conda activate docking_env
python docking/repair_receptors.py --splits train val --workers 12
```


### Output Files

After running the docking script, the following files can be present in each system folder `{SYSTEMS_ROOT}/{split}_systems/{system_id}`, depending on the used settings and docking success:

- `system.cif`: The original PLINDER system file containing all ligands and solvent.
- `system_wo_solvent.cif`: The system file with solvent molecules removed.
- `system_wo_ligand.cif`: The system file with the primary ligand removed.
- `system_sanitized.cif`: Version of the ligand-removed system with sanitized chain/asym IDs (redocking only).
- `system_wo_unsupported.cif`: System without metals unsupported by Meeko. These are converted separately.
- `system_unsupported.cif`: File containing only unsupported atoms removed from the receptor.
- `system_wo_unsupported_FH.cif`: System with unsupported atoms removed and hydrogens optimized with Reduce (if `--add_hydrogen` is enabled).
- `*.pdbqt` files:  
  - `system_unsupported.pdbqt`: PDBQT format of unsupported atoms.  
  - `receptor_merged_<ligand_id>.pdbqt`: **Final prepared receptor file used for docking.**
  - `<ligand_id>.pdbqt`: **Final prepared ligand used for docking.**
- `<ligand_id>.sdf`: Raw ligand file copied from original PLINDER system.
- `cfg.txt`: Configuration file used by AutoDock Vina containing box center and size.
- Vina/Smina: `<method>_<docking_type>_id=<id>_exhaust=<exhaustiveness>_numModes=<num_modes>_eRange=<energy_range>_lig=<ligand_id>.pdbqt`.
- GNINA: `GNINA_<docking_type>_id=<id>_exhaust=<exhaustiveness>_numModes=<num_modes>_prepPH=7.4_lig=<ligand_id>.sdf.gz`.

**Important**: for cross-docking, `system` and `receptor` are replaced with the `alt_system_id`, e.g. `system_wo_solvent.cif` becomes `{alt_system_id}_wo_solvent.cif`.

Finally, the compiled docking results are saved in two CSV files:

- `{SYSTEMS_ROOT}/{split}_systems/{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_{setting}.csv`: Docking results; `{setting}` is `prepPH=7.4` for GNINA and `eRange={energy_range}` for Vina/Smina.
- `{SYSTEMS_ROOT}/{split}_systems/errors_{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_{setting}.csv`: Error logs.

We provide an example notebook for analysing the output files [here](docking/analyse_docking_results.ipynb).

## Pose Re-ranking

### Feature Generation

The features used for re-ranking are computed from the docked poses using
[generate_features.py](docking/generate_features.py). Its default dataset, systems
and output paths are resolved from the repository:

```bash
conda activate docking_env
python docking/generate_features.py --split test \
                            --method GNINA \
                            --type apo_docked_FH \
                            --id 1 \
                            --workers 8
```

The `--split`, `--method`, `--type`, and `--id` arguments must match those of the corresponding `run_docking.py` run, since they are used to locate the docked pose files in `{SYSTEMS_ROOT}/{split}_systems/{system_id}`. Note that `--type` corresponds to `run_docking.py`'s docking type rather than its `--redocking` flag: it is `redocked` or `apo_docked`, with a `_FH` suffix if docking was run with `--add_hydrogen` (the default).

The remaining arguments are:

- `--workers`: number of worker processes (default: `1`, i.e. sequential).
- `--max-systems`: if `> 0`, process only the first N unique systems. Useful for checking a run end to end before committing to the full dataset.
- `--exhaustiveness`, `--num-modes`, `--energy-range`: the settings of the docking run; the energy range applies to Vina/Smina only (defaults match `run_docking.py`).
- `--systems-root`: directory holding the `{split}_systems` folders (default: `$SYSTEMS_ROOT`, or `docking/` itself when unset).
- `--output-dir`: directory for the feature CSV (default: `features`, i.e. `docking/features/`).
- `--dataset-csv`, `--repr-csv`, `--metal-ligand-data`: paths to the curated dataset files produced by [generate_dataset.py](data_curation/generate_dataset.py). These default to the values in [docking_params.py](docking/docking_params.py) and only need to be set if you generated the dataset with non-default parameters.

For the `train` split, only the cluster representatives listed in `--repr-csv` are processed; the other splits are processed in full. Progress is printed per system-ligand pair, and the number of failures is reported at the end.

The result is written to a single CSV:

- `docking/features/features_{split}_{method}_{type}_id={id}.csv`

Each row is one docked pose, identified by `system_id`, `ligand_id`, `split`, `method`, and `pose` (the 1-based rank of the pose within the docked ensemble). The RMSD columns are `symm_rmsd` for the whole ligand and `close_rmsd_3.5_hetero` for the metal-binding group (the CHELATE training target). The remaining columns contain docking scores and generated features: distance-based contact counts and statistics between ligand and receptor atom subsets, Gasteiger charge statistics around the metal, coordination-number decompositions, and donor-metal-donor angle statistics. See the docstrings in [generate_features.py](docking/generate_features.py) for the full feature definitions.

### Model Training

[train_model.py](model/train_model.py) trains one CatBoostRegressor per docking program with Optuna hyperparameter optimisation. The objective is the improvement in metal-binding-group (MBG) top-1 success rate over the docking-score baseline on the validation set. Training samples are weighted by `exp(-k * rmsd^p)`, with `k` and `p` tuned alongside the CatBoost hyperparameters. After optimisation the best model is retrained, with the same weighting, on the combined train+val set. Run it as a module from the repository root:

```bash
conda activate docking_env
python -m model.train_model --method GNINA \
                           --rmsd_cutoff 1.0 \
                           --ntrials 500 \
                           --target close_rmsd_3.5_hetero
```

With no arguments all three methods are trained at the default cutoff. `--method` and `--rmsd_cutoff` both accept several values, training one model per combination. Other arguments of note:

- `--features_dir`: directory holding the feature CSVs from `generate_features.py`.
- `--device` / `--cpu`: select a GPU by index, or force CatBoost onto the CPU.
- `--train_fraction_sweep`: after training, retrain on increasing fractions of the training data and evaluate each on the test set, to show how performance scales with dataset size.

Models and their hyperparameters are written to `model/params/` as `{METHOD}_rmsd={cutoff}_best_model.cbm` and `{METHOD}_rmsd={cutoff}_best_hyperparameters.json`. The cutoff is part of the filename because each cutoff defines a different success-rate objective and therefore a different model. To ship a retrained model with the package, copy its `.cbm` file to `chelate/params/`.

### Test Set Evaluation

[test_eval.py](model/test_eval.py) evaluates trained models on the test set and compares them against the docking programs' own ranking and against the external baselines (AutoDock, GPDOCK, MetalloDock):

```bash
conda activate docking_env
python -m model.test_eval --opt_rmsd_cutoff 1.0
```

The test set is docked several times with identical settings but different run ids. Docking is stochastic, so these repeat runs give different poses; every metric is computed once per run and reported as a mean and standard deviation across runs rather than as a single number. GNINA uses the run ID as its seed: seed 1 for the training and validation poses (run 1) and seeds 1, 2 and 3 for the three test runs. GNINA on the GPU is not bit-for-bit deterministic, so re-running a seed can give slightly different poses; the poses used in the paper are in the Zenodo record.

Cross-docked evaluation reads each system's metals from its prepared apo receptor (`{SYSTEMS_ROOT}/test_systems/{system_id}/{system_id}__{target_system}_merged_{ligand_id}.pdb`, in the Zenodo record under `working_package/docking/test_systems/`); a missing receptor is an error rather than a system without metal features.

`--opt_rmsd_cutoff` selects the model files `{METHOD}_rmsd={cutoff}_best_model.cbm` written by `train_model.py`, read from `--model_dir` (default `model/params/`). `--features_dir` points to the feature CSVs, and `--keep_best_pose` keeps each system's best-predicted pose when none passes the cutoff instead of discarding the system.

Poses without a metal-binding-group label are removed before both the docking-score baseline and CHELATE are evaluated, so both are scored on the same systems. `MBG_sr_top{k}_RMSD={t}` is the success rate among the systems CHELATE retains at a given cutoff; `MBG_sr_all_systems_top{k}_RMSD={t}` counts discarded systems as failures.

Results are written to `model/model_out/` (change this with `--output_dir`), including per-method evaluation results, feature importances and SHAP values, and a set of `plot_repro_*.csv` files holding the exact data behind each figure, indexed by `plot_repro_manifest.json`.

### Metal Identity Probe

[predict_metal_identity.py](model/predict_metal_identity.py) tests whether a trained CHELATE model can distinguish between different metal identities. For every eligible test system the crystal pose is scored once with Vina and featurised, the metal one-hot columns are then overwritten with each metal in `EXPLICIT_METALS` in turn while all other features are held fixed, and the candidate with the lowest predicted MBG RMSD counts as the model's choice. How often that choice matches the crystallographic metal is reported overall and per true metal. Only supports systems with a single metal ion for now.

Run the script as a module from the repository root:

```bash
conda activate docking_env
python -m model.predict_metal_identity --model model/params/VINA_rmsd=1.0_best_model.cbm \
                                      --n-bootstrap 1000 \
                                      --output-dir model/model_out
```

`--model` is the only required argument and takes the path to a trained CatBoost `.cbm` file. Because the crystal pose is rescored with Vina, the model must use the Vina/Smina `docking_score` feature (i.e. not the GNINA model). The remaining arguments are:

- `--dataset-path`, `--metals-json`, `--systems-dir`: input locations, defaulting to the values in [model_params.py](model/model_params.py).
- `--vina-bin`: Vina binary used for the single-point rescoring, defaulting to `VINA_PATH` in [docking_params.py](docking/docking_params.py).
- `--metal-coord-match-tol`: tolerance in Å for matching a metal position from the curated JSON to a metal atom in the receptor PDBQT (default: `2.0`).
- `--max-systems`: process at most this many systems. Every system is rescored with Vina, so a full run is slow; this is intended for checking the pipeline end to end first. The first N systems in dataset order are kept, so the results are not representative.
- `--n-bootstrap`, `--bootstrap-seed`, `--bootstrap-ci`: bootstrap confidence intervals over systems (set `--n-bootstrap 0` to skip).
- `--n-poses`: the crystal pose is featurised as pose 1 of `n_poses`, so `pose_rank_norm = 1 / n_poses` (default: `100`, because the crystal pose is treated as the top pose of a 100-pose docking run, as in training; `1` treats it as the only pose of a single-pose run).
- `--verbose`: print per-system progress.

Results are written to `--output-dir` (default `model/model_out/`):

- `metal_identity_predictions_long.csv`: one row per system and candidate metal, with the predicted MBG RMSD and Vina score.
- `system_prediction_summary.csv`: one row per system, with the true metal, the predicted metal, and whether they match.
- `per_true_metal_accuracy.csv`: recovery accuracy grouped by the true metal.
- `eligible_systems.csv` / `excluded_systems.csv`: which systems entered the probe and why the others did not.
- `system_errors.csv`: systems that failed during processing, with the error message.
- `all_zero_probe_report.csv`: control condition in which the metal is removed from the Vina rescoring and all metal one-hot columns are set to 0 (the geometric features still include the metal), as a reference point for the per-metal predictions.
- `bootstrap_metrics.json` (if bootstrapping is enabled) and `run_summary.json`: aggregate metrics and the run configuration.

## License

CHELATE is released under the [MIT License](LICENSE).
