# ASD Heatmaps

This repository generates and visualizes **percentile heatmaps** of regional brain scores for ASD cohorts (ABCD-ASD & IBIS-ASD). The pipeline used ANTs-based registration transforms to subject-specific heatmaps, computes percentiles, and produces comparison figures.

---

## Table of Contents

1. [Overview](#overview)  
2. [Directory Structure](#directory-structure)  
3. [Requirements](#requirements)  
4. [Environment Setup](#environment-setup)  
5. [Pipeline Workflow](#pipeline-workflow)  
   1. [Step 1: Registered Heatmaps for ABCD-ASD](#step-1-registered-heatmaps-for-abcd-asd)  
   2. [Step 2: Comparison Figures for ABCD-ASD](#step-2-comparison-figures-for-abcd-asd)  
   3. [Step 3: Registered Heatmaps for IBIS-ASD](#step-3-registered-heatmaps-for-ibis-asd)  
6. [Data Outputs](#data-outputs)  
   - [Percentile Images](#percentile-images)  
   - [Output Figures](#output-figures)  
7. [Notebook Descriptions](#notebook-descriptions)  
8. [Usage](#usage)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## Overview

ASD Heatmaps applies previously computed ANTs **SyN** transforms to voxelwise **percentile maps** of ROI outlier scores. For each subject, the pipeline:

1. **Loads** ANTs registration transforms (MNI composite.h5).  
2. **Reads** subject-specific heatmap arrays or NIfTI images.  
3. **Applies** transforms to align data to a common MNI grid.  
4. **Computes** percentile images across inlier reference distributions.  
5. **Saves** and **plots** the processed heatmaps for visual QC and comparisons.

Note: The registered-heatmaps are alread calculated in asd-preprocessing step. See: ..braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/

Supported cohorts:

- **ABCD-ASD** (all samples)  
- **IBIS-ASD**  

---

## Directory Structure

```
asd-heatmaps/
├── registered_heatmaps_abcd_asd.ipynb     ← Step 1 processing for ABCD-ASD  
├── registered_comparison_abcd_asd.ipynb   ← Step 2 comparison plots  
├── registered_heatmaps_ibis_asd.ipynb     ← Step 3 processing for IBIS-ASD  
├── percentile_images_abcd/               ← Generated percentile NIfTIs for ABCD  
├── percentile_images_ibis/               ← Generated percentile NIfTIs for IBIS  
└── output_images/                        ← Final comparison figures (PNG)  
```

---

## Requirements

- Python 3.7+  
- **ANTsPy** (`antspyx`)  
- **NiLearn** (for optional visualization)  
- **NumPy**, **Pandas**, **SciPy**  
- **Matplotlib**, **Seaborn**  
- **Sade** package (for config and transforms)  
- **Jupyter Notebook** or **JupyterLab**

Install via pip:

```bash
pip install antspyx nilearn numpy pandas scipy matplotlib seaborn absl-py ml-collections tqdm
```

---

## Environment Setup

In each notebook, the following environment variables are set:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
project_dir = "../braintypicality"
sys.path.append(project_dir)
```

Adjust GPU index and `project_dir` path as needed.

---

## Pipeline Workflow

### Step 1: Registered Heatmaps for ABCD-ASD

Notebook: `registered_heatmaps_abcd_asd.ipynb`

- **Purpose**: Apply ANTs transforms to original percentile heatmap arrays for ABCD-ASD subjects.  
- **Key steps**:  
  1. Load `transforms_dir` with MNI composite transforms.  
  2. Define `register_to_mni()` to apply transforms to NumPy heatmap arrays or NIfTI channels. (May be already computed in asd-preprocessing/sade_registration and stroed in ..braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/).
  3. Compute percentiles if needed, merge channels, and save aligned images to `percentile_images_abcd/`.

### Step 2: Comparison Figures for ABCD-ASD

Notebook: `registered_comparison_abcd_asd.ipynb`

- **Purpose**: Generate side-by-side visual comparisons of raw vs. percentile maps or T1 vs. T2 channels.  
- **Key steps**:  
  1. Load aligned images from `percentile_images_abcd/`.  
  2. Overlay or tile images using Matplotlib, adding ENS-stat legends.  
  3. Save composite figures to `output_images/`.

### Step 3: Registered Heatmaps for IBIS-ASD

Notebook: `registered_heatmaps_ibis_asd.ipynb`

- **Purpose**: Same process as Step 1, but for the IBIS-cohort.  
- **Key steps**:  
  1. Read IBIS percentile arrays/NIfTIs.  
  2. Apply the same MNI transforms.  
  3. Save outputs to `percentile_images_ibis/`.

---

## Data Outputs

### Percentile Images

- Stored under `percentile_images_abcd/` and `percentile_images_ibis/`.  
- **Format**:  
  - NIfTI (`.nii.gz`) with two channels:  

### Output Figures

- All combined visualization PNGs are saved to `output_images/`.  
- File naming convention:  
  ```
  cohort_subjectID_[comparison|channel]_[...].png
  ```

---

## Notebook Descriptions

- **registered_heatmaps_abcd_asd.ipynb**: apply transforms, save aligned percentile NIfTIs.  
- **registered_comparison_abcd_asd.ipynb**: compare raw vs percentile, T1 vs T2 for ABCD.  
- **registered_heatmaps_ibis_asd.ipynb**: same as first, for IBIS cohort.

---

## Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/USERNAME/ASD-OOD-Analysis.git
   cd ASD-OOD-Analysis/asd-heatmaps
   ```
2. Install requirements.
3. **Step 1**: Open `registered_heatmaps_abcd_asd.ipynb`, run all cells.  
4. **Step 2**: Open `registered_comparison_abcd_asd.ipynb`, run all cells.  
5. **Step 3**: Open `registered_heatmaps_ibis_asd.ipynb`, run all cells.

Responses and figures will populate `percentile_images_*` and `output_images/`.

---

## Contributing

- Please reach out for bugs or feature requests.  

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
