# asd-preprocessing

Scripts for preprocessing and registering brain MRI data for ASD atypicality analysis.  
Built on the “braintypicality-scripts” repository, this pipeline currently supports:

- **ABCD-ASD**  
- **CONTE** & **CONTE-TRIO**  
- **IBIS**, **MSLUB**, **MSSEG**, **HCP**, **BRATS-GLI**, **BRATS-PED**, **EBDS**, **TWINS**, **HCPD**  

---

## Table of Contents

1. [Overview](#overview)  
2. [Directory Structure](#directory-structure)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Step 0 – Generate Identifiers File](#step-0--generate-identifiers-file)  
6. [Step 1 – Preprocessing](#step-1--preprocessing)  
   - [Configuration](#configuration)  
   - [Usage](#usage)  
7. [Step 2 – Registration](#step-2--registration)  
   - [Configuration](#configuration-1)  
   - [Compute Mode](#compute-mode)  
   - [Apply Mode](#apply-mode)  
8. [Step 3 – Visualize Preprocessed Images](#step-3--visualize-preprocessed-images)  
9. [Step 4 – Downstream Analysis](#step-4--downstream-analysis)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## Overview

This repository contains four main workflows:

1. **Generate Identifiers** (`screen_to_identifiers.ipynb`):  
   Extract subject IDs and create cohort-specific text files with identifiers needed for preprocessing.
2. **Preprocessing** (`run_preprocessing_abcd.py`):  
   Brain-extraction, bias correction, histogram matching, and optional segmentation using ANTs & ANTsXNet.
3. **Registration** (`sade_registration_abcd.py`):  
   Compute or apply nonlinear SyN registrations of cropped MRI volumes (and heatmaps) to a common MNI reference.
4. **Visualization** (`display_preprocessed.ipynb`):  
   Inspect original vs. processed T1/T2 images for quality control.
5. **Downstream Analysis**:  
   GHSOM prototype mapping, percentile calculations, and boxplot visualizations. (see: /asd-analysis)

---

## Directory Structure

```
asd-preprocessing/
├── identifier_keys/
│   └── split_passing_keys/
│   └── abcd_qc_passing_keys_160.txt   ← sample QC list
├── ../braintypicality/dataset/        ← Separate directory for keeping the data and cache
│   └── template_cache/                ← ANTs template/cache directory  
├── mri_utils/                         ← helper functions for each cohort  
├── screen_to_identifiers.ipynb        ← Step 0: generate ID lists  
├── run_preprocessing_abcd.py          ← Step 1: main preprocessing script  
├── sade_registration_abcd.py          ← Step 2: compute/apply registrations  
└── display_preprocessed.ipynb         ← Step 3: visualize preprocessed images  
```

---

## Requirements

- Python 3.8+  
- Jupyter / nbconvert (for notebooks)  
- TensorFlow  
- ANTsPy, ANTsPyNet, antsxnet  
- tqdm, numpy, glob, re, concurrent.futures  
- ml_collections, absl, sade.datasets (for registration)  
- matplotlib (for visualization)

Install core dependencies:

```bash
pip install tensorflow ants pytorch-antspynet antsxnet tqdm numpy absl-py ml-collections matplotlib
```

Ensure `mri_utils` and `sade` modules are on your `PYTHONPATH`.

---

## Installation

```bash
git clone https://github.com/VALID_USERNAME/ASD-OOD-Analysis.git
cd ASD-OOD-Analysis/asd-preprocessing
```

---

## Step 0 – Generate Identifiers File

Before preprocessing, extract subject identifiers into text files.  
Open and run `screen_to_identifiers.ipynb` in Jupyter, which will produce, e.g.:

```
abcd_ids.txt
ibis_ids.txt
conte_ids.txt
```
---

## Step 1 – Preprocessing

**Script**: `run_preprocessing_abcd.py`

This script will:

- Read raw T1w/T2w files for the chosen cohort using the generated `<cohort>_ids.txt`.  
- Register to MNI, bias-correct, histogram-match, and optionally segment.  
- Save processed NIfTI volumes (and labels/segmentations) under `DATADIR/Users/<you>/braintyp/...`.

### Configuration

At the top of `run_preprocessing_abcd.py`, edit:

```python
DATADIR  = "/BEE/Connectome/ABCD/"
CACHEDIR = "./dataset/template_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
```

Or override via environment:

```bash
export CUDA_VISIBLE_DEVICES=1
```

### Usage

```bash
python run_preprocessing_abcd.py ABCD
```

Replace `ABCD` with your cohort (e.g., `IBIS`).  
Valid names:  
```
CONTE-TRIO, TWINS, CONTE, MSSEG, CAMCAN, MSLUB,
HCP, BRATS-GLI, BRATS-PED, EBDS, IBIS, HCPD, ABCD
```

---

## Step 2 – Registration

**Script**: `sade_registration_abcd.py`

Supports two modes:

- **compute**: generate SyN transforms.  
- **apply**: apply transforms to raw images or heatmaps.

### Configuration

Edit at top:

```python
CACHE_DIR = "./dataset/template_cache"
DATADIR    = "/BEE/Connectome/ABCD/"
procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"
```

### Compute Mode

For running examples, check the comments at the bottom of the code.

```bash
python sade_registration_abcd.py   --mode compute   --config path/to/biggan_config_abcd_asd.py   --dataset abcd-asd
```

### Apply Mode

```bash
python sade_registration_abcd.py   --mode apply   --config path/to/biggan_config_abcd_asd.py   --dataset abcd-asd   --load_dir path/to/step1/output   --save_dir path/to/registered-images   --image_type original
```

For heatmaps, set `--image_type heatmap`.

---

## Step 3 – Visualize Preprocessed Images

**Notebook**: `display_preprocessed.ipynb`

This notebook loads original and processed volumes and displays slices of T1 and T2:

```python
import ants, os
import matplotlib.pyplot as plt

processed_dir = "/BEE/Connectome/ABCD/Users/emre/braintyp/processed_v3"
original_dir  = "/BEE/Connectome/ABCD/ImageData/Data_abcd_asd_scr_pos_gz"

# List subject IDs
subject_ids = [f[:-7] for f in os.listdir(processed_dir) if f.endswith(".nii.gz")]

# Select a range to view
for subject_id in subject_ids[:10]:
    t1_path = os.path.join(original_dir, f"sub-{subject_id}/ses-baselineYear1Arm1/anat/sub-{subject_id}_T1w.nii.gz")
    t2_path = t1_path.replace("T1w", "T2w")
    proc_path = os.path.join(processed_dir, f"{subject_id}.nii.gz")

    t1 = ants.image_read(t1_path); t2 = ants.image_read(t2_path)
    proc = ants.image_read(proc_path); channels = ants.split_channels(proc)

    print(f"Subject {subject_id}")
    t1.plot(nslices=6); t2.plot(nslices=6)
    channels[0].plot(nslices=6); channels[1].plot(nslices=6)
```

Run in Jupyter or convert to script.

---

## Step 4 – Downstream Analysis

Use provided notebooks for GHSOM prototype mapping, percentile calculations, and boxplot visualizations. Follow tutorials in `asd-analysis/` folder.

---

## Contributing

- Please reach out for bug fixes and feature requests. 

---

## License

MIT License. See [LICENSE](LICENSE).

