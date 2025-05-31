# SADE
**S**core-based **A**nomaly **DE**tection (sah-day)

## Introduction

SADE is a research project aimed at exploring and experimenting with denoising score matching for the purposes of anomaly detection in 3D brain MRIs.

## Installation

To get started with the SADE, follow these installation steps:

1. Clone the repository to your local machine.
2. Set up a Docker environment using the files present in the `docker` directory.
    - Running `build_docker.sh` will build the container and tag it with `<username>/pytorch_sde`
    - The `run_docker.sh` file outlines a potential way to run the container
    - Make sure to mount the local repository where you cloned `sade` into the container
    - Make sure to mount approriate data volumes and use an available port for JupyterLab and Tensorboard
3. You may access the running container using a command such as `docker exec -it <container-name> zsh`
4. (Important) Run `pip install -e .` from the parent directory of `sade`

Note: The user may simply install the requirements in `docker/repo_requirements.txt`. Our model has been tested with Pytorch 2.1. Other library versions can be found in the [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-07.html#rel-23-07) for the NVIDIA container we base the dev image on.

## Usage

One-line quick run
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python main.py --project test --mode train --config configs/ve/toy_config.py --workdir workdir/test/
```
This will go through a dummy training run and save the outputs in a directory `workdir/test/`. Note that th ``--project` parameter is useful for tracking results in Weights and Biases (wandb).

A more involved run could look like:
```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online WANDB_RUN_ID=hres-med \
python main.py --project hres --mode train \
--config configs/ve/single_matryoshka_config.py \
--workdir remote_workdir/hres/frozen-med \
--cuda_opt \
--config.data.dir_path /DATA/processed/ \
--config.data.splits_dir /path/tp/splits/dir/ \
--config.training.batch_size=2 \
--config.eval.batch_size=4 --config.eval.sample_size=2 \
--config.training.log_freq=50 --config.training.eval_freq=100 \
--config.optim.warmup=10_000 --config.optim.lr=1e-5 \
--config.training.sampling_freq=50_000 \
--config.model.trainable_inner_model=False \
--config.data.cache_rate=0
```

Here's a detailed explanation of the training command and its configurations:

### Environment Settings
- `CUDA_VISIBLE_DEVICES=0,1`: Uses GPUs 0 and 1 for data parallel training i.e. the batch will be split across two GPUs
- `WANDB_MODE=online`: Enables online logging to Weights & Biases (will require login at first run)
- `WANDB_RUN_ID=hres-med`: Sets a specific run identifier for tracking


### Core Parameters
- `--project hres`: Project name `hres` for organizing experiments in WandB
- `--mode train`: Sets training mode
- `--config configs/ve/single_matryoshka_config.py`: Uses the Matryoshka model configuration
- `--workdir remote_workdir/hres/frozen-med`: Directory for saving outputs for this experiment - this is expected to be unique for each experiment
- `--cuda_opt`: (Optional) Enables CUDA optimizations - not guaranteed to improve performance

### Training Configuration
- Batch sizes:
  - Training: `--config.training.batch_size=2`
  - Evaluation: `--config.eval.batch_size=4`
  - Sample size: `--config.eval.sample_size=2`
- Frequencies:
  - Logging: `--config.training.log_freq=50`
  - Evaluation: `--config.training.eval_freq=100`
  - Sampling: `--config.training.sampling_freq=50_000`
- Optimization:
  - Warmup steps: `--config.optim.warmup=10_000` - The learning rate is linearly invreased from 0 to `config.optim.lr`. This is common practice for training diffusion models.
  - Learning rate: `--config.optim.lr=1e-5`

### Model Settings
- `--config.model.trainable_inner_model=False`: Freezes inner model weights (specific to matryoshka model)
- `--config.data.cache_rate=0`: Disables data caching

### Data Paths
- `--config.data.dir_path /DATA/processed/`: Path to processed data
- `--config.data.splits_dir /path/tp/splits/dir/`: Path to train/test split files

## Configurations

> [!NOTE]
> It is recommended to store your parameters in config files once they have been finalized after experimentation to help repeatability

The `configs` directory contains configuration files that allow you to customize the project setup. Refer to the individual files within this directory for detailed configuration options and adapt them according to your experimental setups. SADE uses the `ml_collections` library which allows us to easily set configuration parameters in the command line e.g. `--config.eval.batch_size=32`. All configurations are expected to be in the `configs` folder.

An important configuration parameter during evaluation is the `config.eval.experiment`.
The parameters tell the scripts which datasets to use during the evaluation:
- `id`: Experiment identifier
- `train`: Dataset used for training the MSMA-GMM (e.g., "abcd-val")
- `inlier`: Dataset used for inlier evaluation (e.g., "abcd-test")
- `ood`: Dataset or method used for out-of-distribution evaluation (e.g., "tumor", "lesion")

### Training flow models after score model

```bash
WANDB_MODE=online WANDB_RUN_ID="warmed-v0" CUDA_VISIBLE_DEVICES=1 python main.py \
--project flows --mode flow-train --config configs/flows/gmm_flow_config.py \
--workdir /remote/project/experiment/ \
--config.training.pretrained_checkpoint=/path/to/trained-score-model-checkpoint.pth \
--config.flow.patches_per_train_step=16384 --config.flow.patch_batch_size=16384 \
--config.flow.training_kimg=200 \
--config.flow.training_fast_mode=True 
```


## Datasets and Loaders

For our research we used the (ABCD)[https://nda.nih.gov/abcd/] and (HCPD)[https://www.humanconnectome.org/study/hcp-young-adult] datasets. The user will have to request access to these datasets from the linked sources. We do provide notebooks which may be used to generate the train/test splits we used for our training.

### Supporting New Datasets
For training / evaluation on new datsets, make sure to follow the configuration steps below. If you will be training with the provided models, make sure that the dataset can be downsampled by a factor of 8.

#### Configuration Requirements
To add a new dataset, you need to set the following in `config.data`:

  - `dataset`: Name of your new dataset (e.g., "ABCD", "IBIS")
  - `image_size`: Tuple of image dimensions (e.g., (128, 128, 128))
  - `spacing_pix_dim`: Pixel spacing to be used (e.g., 2.0 if you want to downsample by 2)
  - `dir_path`: Path to a flat directory containing the processed data
  - `splits_dir`: Path to the directory containing train/test split files
  - `num_channels`: Number of image channels (e.g. 2 if images have both T1/T2)
  - `cache_rate`: (Optional) Cache rate for dataset (0.0 to 1.0) when loading

> Make sure that `splits_dir` contains the train/val/test filenames as separate text files labeled as `<dataset_name>-<split_name>.txt` e.g. abcd-train.txt. Note that the dataset name is lower case.

The scripts will look at the splits_dir for filenames *not* filepaths, and load them from the `data_dir`. Currently it will look for `.nii.gz` files, although this can be modified in `loaders.py`.

Note that while the configurations can be set at the command line, it may be cleaner to create a new configuration file. Simply copy the `config.ve.biggan_config.py` and update the `config.data` parameters. You may also change the training configurations such as `training.batch_size`.

Our data loaders use MONAI for loading and transforming data. `transforms.py` contains functions to preprocess the data before it is ingested by the network. Feel free to modify and add your own. For instance, we use a `SpatialCrop` which is hard coded for the MNI-152 space and will not be appropriate if you register to some other atlas space.


## Extending the Project

The SADE project is designed with extensibility in mind. Follow these guidelines to add new components to the project:

### Project Structure

The project is structured into several directories and files, each serving a specific purpose:

- `sade`: The main directory containing the core project components.
  - `configs`: Directory storing configuration files for training and evaluation.
  - `datasets`: Directory housing scripts for data preprocessing and loading.
  - `models`: Directory containing scripts defining various architectures and layers.
  - `optim.py`: Script defining optimization strategies for model training.
  - `losses.py`: Script defining score-based loss functions used during training.
  - `sde_lib.py`: Library script containing utilities for the SDEs that determine noise scales.

### Adding New Models

To add new models, create a new file in the `models` directory defining your model. Ensure to register it as done for `ncsnpp3d.py` .

### Adding New Datasets

To incorporate new datasets or data formats, add appropriate data loading to `loaders.py` and any additional preprocessing to `transforms.py` in the `datasets` directory. Make sure to update the configuration files accordingly with the names and image sizes to support the new datasets.

### Testing

To ensure the robustness and reliability of the project, utilize the scripts in the `tests` directory. These scripts provide a framework for testing various components of the project, including data loading, model functionalities, and training procedures.

## License

The project is licensed under the terms mentioned in the `LICENSE` file. Refer to this file for detailed licensing information.
