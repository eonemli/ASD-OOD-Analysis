# asd-analysis

Pipeline for PRADA, ROI‐based GHSOM analysis of ASD cohorts (ABCD‐ASD & IBIS‐ASD).  
This repository covers data processing, GHSOM model training/testing, ROI distributions, correlation analyses, and parcellation studies.

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Data Preparation](#data-preparation)  
   - [Score Normalization](#score-normalization)  
   - [Harmonization](#harmonization)  
   - [Batch Covariates](#batch-covariates)  
4. [GHSOM Model](#ghsom-model)  
   - [Data Structure](#data-structure)  
   - [Training Scripts](#training-scripts)  
   - [Hyperparameters & PCA](#hyperparameters--pca)  
5. [Mapping & Testing](#mapping--testing)  
   - [ABCD‐ASD Test](#abcd-asd-test)  
   - [IBIS‐ASD Test](#ibis-asd-test)  
6. [Parcellation Analysis](#parcellation-analysis)  
7. [ROI Analysis](#roi-analysis)  
   - [Distribution Plots](#distribution-plots)  
   - [Correlation Analysis v1 & v2](#correlation-analysis-v1--v2)  
   - [Likelihood & Classification](#likelihood--classification)  
8. [Outputs & File Formats](#outputs--file-formats)  
9. [Usage Examples](#usage-examples)  
10. [Contributing](#contributing)  
11. [License](#license)  s

---

## Overview

This project analyzes ASD atypicality via **GHSOM** (Growing Hierarchical Self‐Organizing Maps).  
We process ROI scores, train hierarchical SOMs, map outlier subjects, and explore ROI‐prototype relationships through boxplots, correlations, and classification.

Cohorts supported:

- **ABCD‐ASD** (all samples, Philips‐excluded v2 available)  
- **IBIS‐ASD**

---

## Repository Structure

```
asd-analysis/
├── abcd_metadata/               ← Raw clinical & CBCL CSVs + ID keys  
├── spreadsheets/                ← Demographics & cohort spreadsheets  
├── data_scorenorm/              ← Saved score normalization & harmonization  
├── batches/                     ← Scanner/sex/model batch covariates  
├── ghsom_outputs/               ← Trained maps & BMU assignments  
│   ├── trained_maps/            ← Pickled GHSOM objects (.pkl) per hyperparam  
│   └── bmus/                    ← CSVs of best‐matching units per prototype  
├── prada/ghsom_model/           ← GHSOM & GSOM classes + Neuron implementation  
├── train_process_scorenorm_data.ipynb  
├── train_process_harmonization.ipynb  
├── train_process_batches.ipynb  
├── train_ghsom_abcd_inliers_harmonized.py  
├── train_ghsom_abcd_ibis_inliers.py  
├── test_abcd_asd_scorenorm_ghsom.ipynb  
├── test_ibis_asd_scorenorm_ghsom.ipynb  
├── roi_dist_boxplot_ibis_asd_abcd_asd.ipynb  
├── roi_correlation_analysis_abcd_asd_v1.ipynb  
├── roi_correlation_analysis_abcd_asd_v2.ipynb  
├── roi_correlation_analysis_ibis_asd.ipynb  
├── roi_likelihood_harmonization.ipynb  
├── roi_likelihood_classification.ipynb  
├── roi_likelihood_ibis_asd_comparison.ipynb  
└── parcellation_analysis_abcd.ipynb  
```

---

## Data Preparation

### Score Normalization

- **Script**: `train_process_scorenorm_data.ipynb`  
- Loads raw ROI scores, splits into train/val/test sets (`abcd-val`, `abcd-test`, `ibis-inlier`, etc.).  
- Performs z‐score normalization per cohort.  
- Saves `combined_data_score_norms_v*.pkl` and `.npy` in `data_scorenorm/`.

../braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct contains folder for each cohort.
Each cohort folder contains files named with identifiers. Each file contains, orogonal MRI data, heatmaps, scores and score norms.
This file extracts the score norms and saves it for future use, for GHSOM training.

### Harmonization

- **Script**: `train_process_harmonization.ipynb`  
- Applies ComBat to adjust scanner‐ and site‐effects across cohorts.  
- Outputs harmonized score arrays in `data_scorenorm/harmonized_*.npy`.

### Batch Covariates

- **Script**: `train_process_batches.ipynb`  
- Constructs CSVs with batch features (scanner model, sex, etc.) for inclusion in downstream models.  
- Saved under `batches/`.

---

## GHSOM Model

### Data Structure

- **`GHSOM`**: top‐level map class managing hierarchical GSOMs. Similar to tree data structure.
  - **Attributes**:  
    - `.neurons`: dict mapping `(row, col)` → `Neuron` instance.  
    - `.map_shape()`: returns `(n_rows, n_cols)`.  
    - `.weights_map[0]`: 2D array of weight vectors.  
  - **Child Maps**: each `Neuron` has `.child_map` (another `GHSOM` instance) if split occurs.

- **`Neuron`**: unit node in SOM.  
  - **Attributes**:  
    - `.position`: `(row, col)` on map.  
    - `.input_dataset`: samples assigned to this neuron.  
    - `.input_identifiers`: sample identifiers assigned to this neuron to track the samples.
    - `.child_map`: hierarchical child map (or `None`).  
    - `.weight_vector()`, quantization error, activation metrics.

- **GSOM**: underlying Growing SOM implementation used by `GHSOM`. In other words, `GHSOM` is made of hiearchically organized `GSOM`s.

### Training Scripts

- **`train_ghsom_abcd_inliers_harmonized.py`**:  
  - Loads harmonized inlier data (`ibis-inlier`, `abcd-val`, `abcd-test`).  
  - Iterates over log‐spaced `t1`, `t2` parameter grid.  
  - Trains `GHSOM` with `epochs=2000`, `maxiter=10`, `grow_metric="qe"`, saving pickles to `ghsom_outputs/trained_maps/`.

- **`train_ghsom_abcd_ibis_inliers.py`**:  
  - Similar pipeline for IBIS inliers.

### Hyperparameters & PCA

- **PCA**: reduces ROI features to top 5 components prior to SOM training.  
- **GHSOM**:  
  - `learning_rate`, `decay`, `gaussian_sigma`: SOM update params.  
  - `growing_metric="qe"`: quantization‐error threshold.  
  - `t1`, `t2`: expansion criteria.  
  - `epochs=2000`, `maxiter=10`.

---

## Mapping & Testing

### ABCD‐ASD Test

- **Notebook**: `test_abcd_asd_scorenorm_ghsom.ipynb`  
- Loads pickled maps, applies `.winner_neuron()` to ASD outliers.  
- Generates BMU assignments and prototype ranks.

### IBIS‐ASD Test

- **Notebook**: `test_ibis_asd_scorenorm_ghsom.ipynb`  
- Maps IBIS‐ASD samples onto ABCD- and IBIS-trained SOMs for cross‐cohort comparison.

Assigned BMUs saved as CSV in `ghsom_outputs/bmus/`. Dataframes saves as csv files for further analysis.

---

## Parcellation Analysis

- **Notebook**: `parcellation_analysis_abcd.ipynb`  
- Built upon ds-analysis project.
- Examines atlas‐based ROI parcellation effects on atypicality metrics.
- Calculates regional scores and saves them to use in roi_correlation_analysis.

---

## ROI Analysis

### Likelihood & Classification

- **Notebooks**:  
  - `roi_likelihood_harmonization.ipynb`  
  - `roi_likelihood_classification.ipynb`  
  - `roi_likelihood_ibis_asd_comparison.ipynb`  
- Logistic‐regression and classification metrics using ROI signals as predictors.
- Harmonizing the regional scores. 

### Distribution Plots

- **Notebook**: `roi_dist_boxplot_ibis_asd_abcd_asd.ipynb`  
- Boxplots of raw and percentile ROI scores across prototypes side‐by‐side.
- Percentiles are calculated relative to the inlier distributions in each prototype.

### Correlation Analysis v1 & v2

- **Notebooks**:  
  - `roi_correlation_analysis_abcd_asd_v1.ipynb` (all ABCD‐ASD samples).  
  - `roi_correlation_analysis_abcd_asd_v2.ipynb` (excludes Philips‐scanner subjects).  
- Computes Spearman/Pearson correlations between ROI values and prototype rank.
- Rank correlation is used for ROI-Behavior analysis.

- **IBIS**: `roi_correlation_analysis_ibis_asd.ipynb`
Similar notebook that uses IBIS-ASD samples.

---

## Outputs & File Formats

- **Pickled GHSOMs** (`.pkl`): full `GHSOM` objects stored under `ghsom_outputs/trained_maps/`.  
- **BMU CSVs**: mapping of `ID → prototype_rank` in `ghsom_outputs/bmus/`.  
- **Normalized & Harmonized Arrays**: `.npy` & `.pkl` in `data_scorenorm/`.  
- **Batch Covariate CSVs**: scanner/sex/model in `batches/`.  
- **ROI Correlation Results**: CSVs under `roi_behavior_correlations/`.  
- **Spreadsheets**: demographic/clinical CSVs in `spreadsheets/`.  
- **ABCD Metadata**: raw CBCL and clinical scores in `abcd_metadata/`.

---

## Usage Examples

1. **Prepare data**: run notebooks to generate normalized scores, harmonization, and batches.  
2. **Train SOM**:  
   ```bash
   python train_ghsom_abcd_inliers_harmonized.py
   ```  
3. **Test mapping**: open `test_abcd_asd_scorenorm_ghsom.ipynb` and run end‐to‐end.  
4. **ROI analysis**: launch `roi_dist_boxplot_ibis_asd_abcd_asd.ipynb`, then correlation notebooks.

---

## Contributing

- Please reach out for bug fixes and feature requests. 

---

## License

MIT License — see [LICENSE](LICENSE) for details.
