## 1. Fig. 3c headline: top-1 MBG success rate (%; MBG RMSD <= 1 Å), mean ± SD over 3 docking runs

Retained = success among systems with >=1 pose at predicted MBG RMSD <= cutoff. All systems = same CHELATE pick, but systems with no retained pose count as failures. Retained fraction = retained systems / all test systems. All values are on the common 118-pair cohort of all three docking programs, both scenarios and all repeats. Section 7 gives the per-program and all-156 tables.

### Prediction cutoff: operating point (VINA 0.80 Å, SMINA 0.85 Å, GNINA 0.80 Å)

| Dock | Method | Docking top-1 | CHELATE, retained | CHELATE, all systems | Retained fraction |
|---|---|---:|---:|---:|---:|
| crossdocked | Vina | 18.1 ± 0.5 | 32.2 ± 2.7 | 18.6 ± 1.7 | 57.9% (68/118) |
| crossdocked | smina | 18.4 ± 1.3 | 34.0 ± 1.6 | 20.9 ± 0.5 | 61.6% (73/118) |
| crossdocked | GNINA | 18.1 ± 2.1 | 40.2 ± 1.0 | 23.7 ± 0.8 | 59.0% (70/118) |
| redocked | Vina | 31.6 ± 1.3 | 42.5 ± 1.7 | 32.8 ± 1.3 | 77.1% (91/118) |
| redocked | smina | 32.2 ± 1.5 | 45.5 ± 1.2 | 35.3 ± 0.5 | 77.7% (92/118) |
| redocked | GNINA | 33.9 ± 1.7 | 49.3 ± 1.1 | 39.3 ± 1.0 | 79.7% (94/118) |

### Prediction cutoff: 1.0 Å

| Dock | Method | Docking top-1 | CHELATE, retained | CHELATE, all systems | Retained fraction |
|---|---|---:|---:|---:|---:|
| crossdocked | Vina | 18.1 ± 0.5 | 24.2 ± 0.5 | 22.0 ± 0.0 | 91.2% (108/118) |
| crossdocked | smina | 18.4 ± 1.3 | 23.4 ± 0.6 | 20.1 ± 1.0 | 85.6% (101/118) |
| crossdocked | GNINA | 18.1 ± 2.1 | 28.1 ± 1.6 | 25.7 ± 1.3 | 91.5% (108/118) |
| redocked | Vina | 31.6 ± 1.3 | 34.5 ± 1.0 | 33.9 ± 0.8 | 98.3% (116/118) |
| redocked | smina | 32.2 ± 1.5 | 35.5 ± 0.4 | 33.3 ± 0.5 | 93.8% (111/118) |
| redocked | GNINA | 33.9 ± 1.7 | 39.6 ± 1.6 | 38.1 ± 1.5 | 96.3% (114/118) |

External baselines, top-1 MBG success at 1 Å (%):

| Dock | Method | Top-1 |
|---|---|---:|
| crossdocked | AUTODOCK | 12.7 ± 0.8 |
| crossdocked | GPDOCK | 8.5 ± 0.8 |
| crossdocked | METALLODOCK | 31.9 ± 3.0 |
| redocked | AUTODOCK | 21.5 ± 1.3 |
| redocked | GPDOCK | 15.5 ± 0.5 |
| redocked | METALLODOCK | 44.9 ± 2.2 |

## 2. Pearson r, predicted vs true MBG RMSD

Per-system r (Fig. 3a; mean ± SD over systems pooled across runs):

| Dock | Method | r | n systems×runs |
|---|---|---:|---:|
| crossdocked | Vina | 0.401 ± 0.238 | 354 |
| crossdocked | smina | 0.430 ± 0.233 | 354 |
| crossdocked | GNINA | 0.489 ± 0.241 | 354 |
| redocked | Vina | 0.414 ± 0.232 | 354 |
| redocked | smina | 0.435 ± 0.246 | 354 |
| redocked | GNINA | 0.491 ± 0.237 | 354 |

Overall pooled r over all poses (Fig. S16):

| Dock | Method | r |
|---|---|---:|
| crossdocked | Vina | 0.437 |
| crossdocked | smina | 0.436 |
| crossdocked | GNINA | 0.489 |
| redocked | Vina | 0.436 |
| redocked | smina | 0.454 |
| redocked | GNINA | 0.503 |

## 3. Binding-affinity Pearson r (Fig. S20): docking score vs experimental affinity, top-1 pose

Baseline = all systems with affinity; filtered = CHELATE-retained systems. n = systems with affinity (mean over runs).

Prediction cutoff, operating point (VINA 0.80 Å, SMINA 0.85 Å, GNINA 0.80 Å):

| Dock | Method | Baseline (n) | Filtered (n) |
|---|---|---:|---:|
| crossdocked | Vina | 0.172 ± 0.040 (30) | -0.111 ± 0.064 (16) |
| crossdocked | smina | 0.150 ± 0.022 (30) | 0.040 ± 0.020 (16) |
| crossdocked | GNINA | 0.445 ± 0.037 (30) | 0.301 ± 0.122 (17) |
| redocked | Vina | 0.163 ± 0.004 (30) | 0.259 ± 0.016 (21) |
| redocked | smina | 0.138 ± 0.001 (30) | 0.324 ± 0.028 (21) |
| redocked | GNINA | 0.358 ± 0.023 (30) | 0.253 ± 0.114 (22) |

Prediction cutoff, 1.0 Å:

| Dock | Method | Baseline (n) | Filtered (n) |
|---|---|---:|---:|
| crossdocked | Vina | 0.172 ± 0.040 (30) | 0.125 ± 0.013 (26) |
| crossdocked | smina | 0.150 ± 0.022 (30) | 0.053 ± 0.018 (21) |
| crossdocked | GNINA | 0.445 ± 0.037 (30) | 0.599 ± 0.015 (24) |
| redocked | Vina | 0.163 ± 0.004 (30) | 0.216 ± 0.010 (29) |
| redocked | smina | 0.138 ± 0.001 (30) | 0.249 ± 0.054 (27) |
| redocked | GNINA | 0.358 ± 0.023 (30) | 0.301 ± 0.072 (26) |

## 4. Feature importance (Fig. S23 CatBoost PredictionValuesChange; Fig. S24 SHAP), top 5 per model

| Method | Feature importance | |mean SHAP| |
|---|---|---|
| Vina | cn_lig_3.5 (4.1), angle_dev_octahedral (4), ch_4.0_mean (2.3), mbg_charge_mean_5.5 (2), mbg_charge_mean_3.5 (2) | cn_lig_3.5 (0.063), angle_dev_octahedral (0.051), ch_4.0_mean (0.04), ch_4.0_min (0.029), ch_4.5_mean (0.029) |
| smina | cn_lig_3.5 (5.3), ch_4.5_mean (4.6), angle_dev_octahedral (4), mbg_charge_mean_5.5 (2.4), ch_3.0_min (2.2) | angle_dev_octahedral (0.054), cn_lig_3.5 (0.053), ch_4.5_mean (0.04), pose_rank_norm (0.031), mbg_charge_mean_5.5 (0.031) |
| GNINA | ch_4.0_mean (4.8), angle_dev_octahedral (4.3), cn_lig_3.5 (3.5), ch_4.5_mean (2.8), mbg_charge_mean_5.5 (2.4) | angle_dev_octahedral (0.051), cn_lig_3.5 (0.047), ch_4.0_mean (0.043), mbg_charge_mean_3.5 (0.038), mbg_charge_mean_5.5 (0.031) |

## 5. Fig. S22 subsets, cross-docked, prediction cutoff VINA 0.80 Å, SMINA 0.85 Å, GNINA 0.80 Å, MBG SR (%) baseline → CHELATE (retained)

| Subset | Method | MBG SR |
|---|---|---|
| Enzymes | Vina | 17.5 → 37.7 (n 105 → 56/61/58) |
| Enzymes | smina | 17.8 → 38.8 (n 105 → 62/64/65) |
| Enzymes | GNINA | 19.4 → 46.1 (n 105 → 58/63/59) |
| Non-enzymes | Vina | 23.1 → 0.0 (n 13 → 10/10/10) |
| Non-enzymes | smina | 23.1 → 0.0 (n 13 → 8/10/9) |
| Non-enzymes | GNINA | 7.7 → 3.3 (n 13 → 10/9/10) |
| Zn | Vina | 9.6 → 37.4 (n 45 → 21/25/21) |
| Zn | smina | 9.6 → 39.4 (n 45 → 20/23/23) |
| Zn | GNINA | 17.0 → 43.6 (n 45 → 22/23/19) |
| Mg | Vina | 35.1 → 27.1 (n 19 → 16/16/16) |
| Mg | smina | 36.8 → 32.1 (n 19 → 14/17/16) |
| Mg | GNINA | 26.3 → 43.8 (n 19 → 16/16/16) |
| Fe | Vina | 25.0 → 55.6 (n 12 → 4/4/3) |
| Fe | smina | 25.0 → 35.6 (n 12 → 6/6/5) |
| Fe | GNINA | 25.0 → 61.1 (n 12 → 3/3/4) |
| Ca | Vina | 20.0 → 22.0 (n 15 → 14/13/14) |
| Ca | smina | 17.8 → 19.0 (n 15 → 14/14/14) |
| Ca | GNINA | 11.1 → 22.2 (n 15 → 15/15/15) |
| Ni | Vina | 23.3 → 25.0 (n 10 → 2/3/4) |
| Ni | smina | 23.3 → 11.1 (n 10 → 3/3/2) |
| Ni | GNINA | 10.0 → 11.1 (n 10 → 1/2/3) |

## 6. Training-fraction sweep (Fig. S28)

Fixed final hyperparameters, deterministic training-group subsets; no Optuna retuning. Success is MBG RMSD <= 1 Å among retained systems. Values are mean ± sample SD over the three test docking runs. Fractions refer to the available train+validation groups; the final models use 100%.

Input: `mbg_sr_vs_training_fraction_predCutoff=VINA-0p8_SMINA-0p85_GNINA-0p8.csv`.

| Engine | Test | Training fraction | Cutoff (Å) | Groups used / total | Retained systems (mean) | MBG success (%) |
|---|---|---:|---:|---:|---:|---:|
| GNINA | crossdocked | 10% | 0.80 | 395 / 3950 | 36.0 | 47.2 ± 1.6 |
| GNINA | crossdocked | 20% | 0.80 | 790 / 3950 | 53.3 | 40.6 ± 2.7 |
| GNINA | crossdocked | 30% | 0.80 | 1185 / 3950 | 63.3 | 40.5 ± 2.1 |
| GNINA | crossdocked | 40% | 0.80 | 1580 / 3950 | 55.7 | 43.7 ± 7.9 |
| GNINA | crossdocked | 50% | 0.80 | 1975 / 3950 | 65.0 | 38.4 ± 4.0 |
| GNINA | crossdocked | 60% | 0.80 | 2370 / 3950 | 66.7 | 39.6 ± 3.1 |
| GNINA | crossdocked | 70% | 0.80 | 2765 / 3950 | 70.3 | 39.4 ± 4.1 |
| GNINA | crossdocked | 80% | 0.80 | 3160 / 3950 | 72.3 | 40.5 ± 1.5 |
| GNINA | crossdocked | 90% | 0.80 | 3555 / 3950 | 69.3 | 38.5 ± 4.0 |
| GNINA | crossdocked | 100% | 0.80 | 3950 / 3950 | 69.7 | 40.2 ± 1.0 |
| GNINA | redocked | 10% | 0.80 | 395 / 3950 | 57.0 | 53.1 ± 3.1 |
| GNINA | redocked | 20% | 0.80 | 790 / 3950 | 79.0 | 49.8 ± 1.0 |
| GNINA | redocked | 30% | 0.80 | 1185 / 3950 | 85.3 | 53.6 ± 3.4 |
| GNINA | redocked | 40% | 0.80 | 1580 / 3950 | 79.0 | 55.7 ± 0.8 |
| GNINA | redocked | 50% | 0.80 | 1975 / 3950 | 91.0 | 52.0 ± 1.7 |
| GNINA | redocked | 60% | 0.80 | 2370 / 3950 | 91.0 | 53.5 ± 0.9 |
| GNINA | redocked | 70% | 0.80 | 2765 / 3950 | 91.7 | 51.3 ± 1.9 |
| GNINA | redocked | 80% | 0.80 | 3160 / 3950 | 91.7 | 51.3 ± 0.8 |
| GNINA | redocked | 90% | 0.80 | 3555 / 3950 | 92.0 | 51.1 ± 1.1 |
| GNINA | redocked | 100% | 0.80 | 3950 / 3950 | 94.0 | 49.3 ± 1.1 |
| SMINA | crossdocked | 10% | 0.85 | 370 / 3691 | 46.3 | 40.5 ± 4.1 |
| SMINA | crossdocked | 20% | 0.85 | 739 / 3691 | 61.7 | 31.4 ± 0.3 |
| SMINA | crossdocked | 30% | 0.85 | 1108 / 3691 | 67.3 | 34.2 ± 3.4 |
| SMINA | crossdocked | 40% | 0.85 | 1477 / 3691 | 64.0 | 35.0 ± 2.3 |
| SMINA | crossdocked | 50% | 0.85 | 1846 / 3691 | 67.3 | 36.3 ± 3.5 |
| SMINA | crossdocked | 60% | 0.85 | 2215 / 3691 | 68.7 | 31.2 ± 3.2 |
| SMINA | crossdocked | 70% | 0.85 | 2584 / 3691 | 73.3 | 33.7 ± 1.4 |
| SMINA | crossdocked | 80% | 0.85 | 2953 / 3691 | 67.7 | 36.1 ± 3.4 |
| SMINA | crossdocked | 90% | 0.85 | 3322 / 3691 | 67.7 | 35.1 ± 3.3 |
| SMINA | crossdocked | 100% | 0.85 | 3691 / 3691 | 72.7 | 34.0 ± 1.6 |
| SMINA | redocked | 10% | 0.85 | 370 / 3691 | 62.7 | 49.5 ± 3.3 |
| SMINA | redocked | 20% | 0.85 | 739 / 3691 | 80.3 | 50.2 ± 2.2 |
| SMINA | redocked | 30% | 0.85 | 1108 / 3691 | 89.7 | 43.1 ± 2.3 |
| SMINA | redocked | 40% | 0.85 | 1477 / 3691 | 83.7 | 47.8 ± 2.0 |
| SMINA | redocked | 50% | 0.85 | 1846 / 3691 | 89.3 | 46.6 ± 0.9 |
| SMINA | redocked | 60% | 0.85 | 2215 / 3691 | 93.0 | 44.8 ± 1.6 |
| SMINA | redocked | 70% | 0.85 | 2584 / 3691 | 95.3 | 45.5 ± 1.2 |
| SMINA | redocked | 80% | 0.85 | 2953 / 3691 | 89.3 | 47.0 ± 1.2 |
| SMINA | redocked | 90% | 0.85 | 3322 / 3691 | 92.3 | 44.8 ± 2.6 |
| SMINA | redocked | 100% | 0.85 | 3691 / 3691 | 91.7 | 45.5 ± 1.2 |
| VINA | crossdocked | 10% | 0.80 | 375 / 3743 | 40.7 | 34.3 ± 5.9 |
| VINA | crossdocked | 20% | 0.80 | 749 / 3743 | 47.3 | 38.7 ± 3.7 |
| VINA | crossdocked | 30% | 0.80 | 1123 / 3743 | 64.7 | 28.9 ± 2.6 |
| VINA | crossdocked | 40% | 0.80 | 1498 / 3743 | 63.7 | 34.0 ± 6.2 |
| VINA | crossdocked | 50% | 0.80 | 1872 / 3743 | 65.3 | 33.6 ± 3.9 |
| VINA | crossdocked | 60% | 0.80 | 2246 / 3743 | 61.7 | 32.3 ± 1.8 |
| VINA | crossdocked | 70% | 0.80 | 2621 / 3743 | 66.3 | 34.2 ± 0.3 |
| VINA | crossdocked | 80% | 0.80 | 2995 / 3743 | 71.0 | 31.5 ± 0.8 |
| VINA | crossdocked | 90% | 0.80 | 3369 / 3743 | 71.3 | 30.8 ± 0.4 |
| VINA | crossdocked | 100% | 0.80 | 3743 / 3743 | 68.3 | 32.2 ± 2.7 |
| VINA | redocked | 10% | 0.80 | 375 / 3743 | 50.3 | 50.4 ± 2.0 |
| VINA | redocked | 20% | 0.80 | 749 / 3743 | 66.0 | 47.1 ± 4.5 |
| VINA | redocked | 30% | 0.80 | 1123 / 3743 | 87.3 | 44.4 ± 2.6 |
| VINA | redocked | 40% | 0.80 | 1498 / 3743 | 86.3 | 42.9 ± 0.7 |
| VINA | redocked | 50% | 0.80 | 1872 / 3743 | 87.3 | 45.9 ± 3.1 |
| VINA | redocked | 60% | 0.80 | 2246 / 3743 | 86.0 | 41.0 ± 3.5 |
| VINA | redocked | 70% | 0.80 | 2621 / 3743 | 84.7 | 49.6 ± 2.2 |
| VINA | redocked | 80% | 0.80 | 2995 / 3743 | 89.0 | 44.9 ± 2.9 |
| VINA | redocked | 90% | 0.80 | 3369 / 3743 | 90.0 | 45.9 ± 2.1 |
| VINA | redocked | 100% | 0.80 | 3743 / 3743 | 91.0 | 42.5 ± 1.7 |

## 7. Cohorts: top-1 MBG ≤1 Å and whole-molecule RMSD ≤2 Å

Means over three repeats. Overall success uses the declared denominator; unavailable or rejected pairs count as failures. Retained success uses retained pairs. The common cohort is fixed by input/label availability across all engines, scenarios and repeats. Engine-evaluable populations can differ. The curated population contains all 156 pairs.

| Cohort | Dock | Method | Prediction cutoff (Å) | Denominator | Eligible | Retained | Coverage (%) | Native MBG (%) | CHELATE MBG overall (%) | CHELATE MBG retained (%) | Native whole (%) | CHELATE whole overall (%) | CHELATE whole retained (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| common_evaluable | redocked | VINA | 0.80 | 118.00 | 118.00 | 91.00 | 77.12 | 31.64 | 32.77 | 42.51 | 39.27 | 27.97 | 36.33 |
| common_evaluable | redocked | SMINA | 0.85 | 118.00 | 118.00 | 91.67 | 77.68 | 32.20 | 35.31 | 45.46 | 38.14 | 34.75 | 44.74 |
| common_evaluable | redocked | GNINA | 0.80 | 118.00 | 118.00 | 94.00 | 79.66 | 33.90 | 39.27 | 49.29 | 49.44 | 42.37 | 53.15 |
| common_evaluable | crossdocked | VINA | 0.80 | 118.00 | 118.00 | 68.33 | 57.91 | 18.08 | 18.64 | 32.19 | 15.54 | 11.02 | 18.99 |
| common_evaluable | crossdocked | SMINA | 0.85 | 118.00 | 118.00 | 72.67 | 61.58 | 18.36 | 20.90 | 33.98 | 16.10 | 13.56 | 22.06 |
| common_evaluable | crossdocked | GNINA | 0.80 | 118.00 | 118.00 | 69.67 | 59.04 | 18.08 | 23.73 | 40.19 | 24.58 | 20.06 | 34.02 |
| engine_evaluable | redocked | VINA | 0.80 | 137.00 | 137.00 | 105.33 | 76.89 | 30.41 | 31.87 | 41.48 | 40.63 | 27.01 | 35.16 |
| engine_evaluable | redocked | SMINA | 0.85 | 135.00 | 135.00 | 104.00 | 77.04 | 31.85 | 33.09 | 42.95 | 40.00 | 33.83 | 43.91 |
| engine_evaluable | redocked | GNINA | 0.80 | 141.00 | 141.00 | 115.00 | 81.56 | 32.62 | 37.12 | 45.52 | 49.65 | 40.66 | 49.82 |
| engine_evaluable | crossdocked | VINA | 0.80 | 130.00 | 130.00 | 76.00 | 58.46 | 18.97 | 18.97 | 32.45 | 14.87 | 10.00 | 17.08 |
| engine_evaluable | crossdocked | SMINA | 0.85 | 128.00 | 128.00 | 78.00 | 60.94 | 18.49 | 20.05 | 32.94 | 15.62 | 12.50 | 20.57 |
| engine_evaluable | crossdocked | GNINA | 0.80 | 132.00 | 132.00 | 79.33 | 60.10 | 18.18 | 23.23 | 38.65 | 24.24 | 20.45 | 34.06 |
| curated_156 | redocked | VINA | 0.80 | 156.00 | 137.00 | 105.33 | 67.52 | 26.71 | 27.99 | 41.48 | 35.68 | 23.72 | 35.16 |
| curated_156 | redocked | SMINA | 0.85 | 156.00 | 135.00 | 104.00 | 66.67 | 27.56 | 28.63 | 42.95 | 34.62 | 29.27 | 43.91 |
| curated_156 | redocked | GNINA | 0.80 | 156.00 | 141.00 | 115.00 | 73.72 | 29.49 | 33.55 | 45.52 | 44.87 | 36.75 | 49.82 |
| curated_156 | crossdocked | VINA | 0.80 | 156.00 | 130.00 | 76.00 | 48.72 | 15.81 | 15.81 | 32.45 | 12.39 | 8.33 | 17.08 |
| curated_156 | crossdocked | SMINA | 0.85 | 156.00 | 128.00 | 78.00 | 50.00 | 15.17 | 16.45 | 32.94 | 12.82 | 10.26 | 20.57 |
| curated_156 | crossdocked | GNINA | 0.80 | 156.00 | 132.00 | 79.33 | 50.85 | 15.38 | 19.66 | 38.65 | 20.51 | 17.31 | 34.06 |

External baselines on both declared populations; missing outputs and invalid labels count as failures.

| Cohort | Dock | Method | Denominator | Available outputs | MBG ≤1 Å (%) | Whole RMSD ≤2 Å (%) |
|---|---|---|---:|---:|---:|---:|
| common_evaluable | crossdocked | AUTODOCK | 118 | 96.00 | 12.71 | 12.15 |
| common_evaluable | crossdocked | GPDOCK | 118 | 85.33 | 8.47 | 20.62 |
| common_evaluable | crossdocked | METALLODOCK | 118 | 114.00 | 31.92 | 30.51 |
| common_evaluable | redocked | AUTODOCK | 118 | 96.00 | 21.47 | 24.86 |
| common_evaluable | redocked | GPDOCK | 118 | 96.67 | 15.54 | 31.64 |
| common_evaluable | redocked | METALLODOCK | 118 | 117.00 | 44.92 | 44.07 |
| curated_156 | crossdocked | AUTODOCK | 156 | 125.00 | 10.90 | 9.83 |
| curated_156 | crossdocked | GPDOCK | 156 | 104.33 | 7.69 | 18.16 |
| curated_156 | crossdocked | METALLODOCK | 156 | 145.33 | 27.35 | 29.06 |
| curated_156 | redocked | AUTODOCK | 156 | 126.00 | 19.87 | 23.93 |
| curated_156 | redocked | GPDOCK | 156 | 115.67 | 14.32 | 28.63 |
| curated_156 | redocked | METALLODOCK | 156 | 146.33 | 41.03 | 41.03 |
