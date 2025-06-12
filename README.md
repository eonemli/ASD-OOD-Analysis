# Phenotype Representation and Analysis via Discriminative Atypicality (PRADA)

A monorepo of coordinated pipelines for preprocessing, modeling, and analysis of ASD (Autism Spectrum Disorder) imaging data, plus Conte‐cohort analyses and deep generative modeling support.

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Cloning & Setup](#cloning--setup)  
   - [Workflow Overview](#workflow-overview)  
4. [Subprojects](#subprojects)  
   1. [asd-preprocessing](#asd-preprocessing)  
   2. [asd-analysis](#asd-analysis)  
   3. [asd-heatmaps](#asd-heatmaps)  
   4. [conte-analysis](#conte-analysis)  
   5. [sade](#sade)  
5. [Contributing](#contributing)  
6. [License](#license)  

---

## Overview

This repository organizes a full pipeline for ASD neuroimaging work:

- **Preprocessing** raw MRI (brain extraction, registration, segmentation)  
- **GHSOM‐based analysis** of ROI distributions, correlations, and classification  
- **Heatmap generation**: percentile‐based voxelwise maps aligned to MNI  
- **Conte‐cohort analysis** (under development)  
- **Sade** submodule for deep generative models and data transforms (spatial-MSMA framework)

You can run the pipeline end‐to‐end or pick individual modules.

---

## Repository Structure

```
.
├── asd-preprocessing/   ← Step 1: raw MRI preprocessing pipelines
├── asd-analysis/        ← Step 2: GHSOM training, ROI‐based analysis
├── asd-heatmaps/        ← Step 3: generate & visualize percentile heatmaps
├── conte-analysis/      ← Step 4: Conte cohort analysis (under development)
└── sade/                ← Deep generative “Sade” framework & utilities
```

Each folder is a standalone project with its own `README.md`, scripts, notebooks, and dependencies.

---

## Getting Started

### Prerequisites

- **Python** 3.7+ (3.8+ recommended)  
- **Git**, **Conda** or **venv**  
- Common libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`  
- Imaging libraries (as needed): `tensorflow`, `ants`, `antspynet`, `antsxnet`, `sade`, `ml_collections`, etc.

### Cloning & Setup

```bash
git clone https://github.com/USERNAME/ASD-OOD-Analysis.git
cd ASD-OOD-Analysis
```

Create & activate an environment:

```bash
conda create -n asd-env python=3.8
conda activate asd-env
pip install numpy pandas matplotlib seaborn tqdm antspyx ml-collections absl-py
```

Each subproject may require additional installs—see its `README.md`.

### Workflow Overview

Recommended order:

1. **`sade/`**              – deep generative experiments and utilities (needs to be installed before running other scripts)  
2. **`asd-preprocessing/`** – prepare raw MRI for analysis  
3. **`asd-analysis/`**      – train/test GHSOMs, ROI distributions, correlations  
4. **`asd-heatmaps/`**      – generate percentile heatmaps, save aligned NIfTIs  
5. **`conte-analysis/`**    – Conte cohort analysis (under development)  

---

## Subprojects

### `sade`

- **Purpose**: Contains the Sade deep generative modeling framework—data loaders, model configs, transforms.  
- **Key files**:  
  - `sade/` Python package  
- **See**: `sade/README.md`.

### `asd-preprocessing`

- **Purpose**: Brain‐extraction, bias correction, ANTs registration, segmentation.  
- **Key files**:  
  - `run_preprocessing_abcd.py`  
  - Notebooks: `screen_to_identifiers.ipynb`, `display_preprocessed.ipynb`  
- **See**: `asd-preprocessing/README.md`.

### `asd-analysis`

- **Purpose**:  
  - Normalize & harmonize ROI scores  
  - Train hierarchical GHSOMs (Growing Hierarchical SOMs)  
  - Map ASD outliers to prototypes  
  - ROI‐based boxplots, correlations, classification  
- **Key files**:  
  - Score normalization & harmonization notebooks  
  - `train_ghsom_abcd_inliers_harmonized.py`, `train_ghsom_abcd_ibis_inliers.py`  
  - ROI analysis notebooks: `roi_dist_boxplot_*.ipynb`, `roi_correlation_analysis_*.ipynb`, `roi_likelihood_*.ipynb`  
- **See**: `asd-analysis/README.md`.

### `asd-heatmaps`

- **Purpose**: Apply ANTs SyN transforms to subject‐level ROI percentile arrays, produce aligned heatmaps and comparison figures.  
- **Key files**:  
  - `registered_heatmaps_abcd_asd.ipynb`, `registered_comparison_abcd_asd.ipynb`, `registered_heatmaps_ibis_asd.ipynb`  
- **See**: `asd-heatmaps/README.md`.

### `conte-analysis`

- **Purpose**: Analogous GHSOM & ROI analysis for the Conte cohort.  
- **Status**: Under development and notebooks will mirror `asd-analysis`.  
- **See**: `conte-analysis/README.md`.

---

## Contributing

We welcome contributions!

1. **Fork** this repository  
2. **Branch**: `git checkout -b feature/xyz`  
3. **Commit** with clear messages  
4. **Push** and **open a Pull Request**  

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
