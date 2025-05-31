import glob
import os
import re
import sys

import ants
import numpy as np
from absl import app, flags
from ml_collections.config_flags import config_flags
from sade.datasets.loaders import get_image_files_list
from sade.datasets.transforms import get_val_transform
from tqdm import tqdm

# CACHE_DIR = "/codespace/braintypicality/dataset/template_cache/"
CACHE_DIR = "/ASD2/emre_projects/OOD/braintypicality2/braintypicality/dataset/template_cache"
DATADIR = "/BEE/Connectome/ABCD/"
# DATADIR = "/DATA/"
procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"

####
if not os.path.exists(procd_ref_img_path):
    T1_REF_IMG_PATH = os.path.join(
        CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t1_tal_nlin_sym_09a.nrrd"
    )
    T2_REF_IMG_PATH = os.path.join(
        CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t2_tal_nlin_sym_09a.nrrd"
    )
    MASK_REF_IMG_PATH = os.path.join(
        CACHE_DIR, "mni_icbm152_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nrrd"
    )

    ants_mni = ants.image_read(f"{CACHE_DIR}/croppedMni152.nii.gz")
    t1_ref_img = ants.image_read(T1_REF_IMG_PATH)
    t2_ref_img = ants.image_read(T2_REF_IMG_PATH)
    ref_img_mask = ants.image_read(MASK_REF_IMG_PATH)

    # Use ANTs' tighter cropping
    diff = np.array(t1_ref_img.shape) - np.array(ants_mni.shape)
    crop_idxs_start, crop_idxs_end = (
        1 + diff // 2,
        np.array(t1_ref_img.shape) - diff // 2,
    )

    t1_ref_img = ants.crop_indices(t1_ref_img, crop_idxs_start, crop_idxs_end)
    t2_ref_img = ants.crop_indices(t2_ref_img, crop_idxs_start, crop_idxs_end)
    ref_img_mask = ants.crop_indices(ref_img_mask, crop_idxs_start, crop_idxs_end)

    procd_ref_img = ants.merge_channels(
        (t1_ref_img * ref_img_mask, t2_ref_img * ref_img_mask)
    )

    procd_ref_img.to_filename(procd_ref_img_path)
####


def run_registration(config, dataset_name):

    # savedir = f"/{DATADIR}/Users/amahmood/braintyp/spacing_{int(config.data.spacing_pix_dim)}-ants"
    savedir = f"/{DATADIR}/Users/emre/braintyp/spacing_{int(config.data.spacing_pix_dim)}-ants"
    os.makedirs(savedir, exist_ok=True)

    img_loader = get_val_transform(config)
    procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"
    ref_img_tensor = img_loader({"image": procd_ref_img_path})["image"].numpy()
    ref_img_post_transform = (ants.from_numpy(ref_img_tensor[0]) + 1) / 2

    fnames = get_image_files_list(
        dataset_name, config.data.dir_path, config.data.splits_dir
    )

    for fname_dict in tqdm(fnames):
        img_tensor = img_loader(fname_dict)["image"].numpy()
        t1_img = (ants.from_numpy(img_tensor[0]) + 1) / 2

        fname = fname_dict["image"]
        sampleid = os.path.basename(fname).split(".nii.gz")[0]

        if config.data.spacing_pix_dim == 1.0:
            # reg_iterations = [40, 20, 10, 0]
            reg_iterations = [500, 80, 40, 0]
        else:
            reg_iterations = [40, 20, 0]

        _ = ants.registration(
            fixed=ref_img_post_transform,
            moving=t1_img,
            # More info at https://github.com/ANTsX/ANTs/blob/master/Scripts/antsRegistrationSyN.sh
            type_of_transform="antsRegistrationSyN[s]",
            outprefix=f"{savedir}/{sampleid}",
            write_composite_transform=True,
            reg_iterations=reg_iterations,
            verbose=0,
        )


def extract_id(fname):
    return re.match("(.*)(.nii.gz|.npz)$", fname).group(1)


def apply_registration(config, load_dir, save_dir, image_type):
    os.makedirs(save_dir, exist_ok=True)
    transforms_dir = f"/{DATADIR}/Users/amahmood/braintyp/spacing_{int(config.data.spacing_pix_dim)}"
    #transforms_dir = f"/{DATADIR}/Users/emre/braintyp/spacing_{int(config.data.spacing_pix_dim)}-ants"
    procd_ref_img_path = f"{CACHE_DIR}/cropped_niral_mni.nii.gz"

    img_loader = get_val_transform(config)
    ref_img_tensor = img_loader({"image": procd_ref_img_path})["image"].numpy()
    ref_img_post_transform = (ants.from_numpy(ref_img_tensor[0]) + 1) / 2

    print(
        f"Applying registrations from {transforms_dir} to images in {load_dir} and saving to {save_dir}."
    )
    paths = glob.glob(os.path.join(load_dir, "*"))
    for fname in tqdm(paths):

        if ".npz" in fname:
            x = np.load(fname)["heatmap"] if image_type == "heatmap" else np.load(fname)["original"]
            x = ants.from_numpy(x) if image_type == "heatmap" else [ants.from_numpy((x[c])) for c in range(x.shape[0])]
        elif ".nii.gz" in fname:
            x = ants.image_read(fname)
        else:
            raise ValueError(
                f"File format of {fname} not recognized - must be .nii.gz or .npz."
            )

        sid = extract_id(os.path.basename(fname))
        transform_mat = f"{transforms_dir}/{sid}Composite.h5"
        
        if image_type == "original":
            ch1, ch2 = x[0], x[1]

            h_aligned_1 = ants.apply_transforms(
                fixed=ref_img_post_transform,
                interpolator="linear",
                verbose=False,
                moving=ch1,
                transformlist=transform_mat,
            )

            h_aligned_2 = ants.apply_transforms(
                fixed=ref_img_post_transform,
                interpolator="linear",
                verbose=False,
                moving=ch2,
                transformlist=transform_mat,
            )
            h_aligned = ants.merge_channels([h_aligned_1, h_aligned_2])

        else:
            h_aligned = ants.apply_transforms(
                fixed=ref_img_post_transform,
                interpolator="linear",
                verbose=False,
                moving=x,
                transformlist=transform_mat,
            )

        h_aligned.to_filename(f"{save_dir}/{sid}.nii.gz")


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Sade configuration used to train the model.", lock_config=True
)
flags.DEFINE_enum(
    "mode",
    None,
    ["compute", "apply"],
    "Whether to compute the registrations and save them or apply them from saved directory.",
)
flags.DEFINE_string(
    "dataset",
    None,
    "The dataset for which to run the registration (should match names used in config and splits dir).",
)
flags.DEFINE_string(
    "load_dir",
    None,
    "Directory containing the images to register - only used in apply mode.",
    required=False,
)
flags.DEFINE_string(
    "save_dir",
    None,
    "Directory to save the registered images - only used in apply mode.",
    required=False,
)
flags.DEFINE_enum(
    "image_type",
    None,
    ["original", "heatmap"],
    "Whether to apply the registration to the original images or the heatmaps - only used in apply mode.",
    required=False,
)


def main(argv):
    if FLAGS.mode == "compute":
        config = FLAGS.config
        config.data.splits_dir = (
            #"/ASD/ahsan_projects/Developer/braintypicality-scripts/split-keys/"
            "/ASD2/emre_projects/OOD/scripts/braintypicality-scripts/split-keys" # additional abcd-asd_keys.txt file
        )
        run_registration(FLAGS.config, FLAGS.dataset)
    elif FLAGS.mode == "apply":
        assert FLAGS.load_dir is not None, "load_dir must be provided in apply mode."
        assert FLAGS.save_dir is not None, "save_dir must be provided in apply mode."
        assert FLAGS.image_type is not None, "image_type (original or heatmap) must be provided in apply mode."
        apply_registration(FLAGS.config, FLAGS.load_dir, FLAGS.save_dir, FLAGS.image_type)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


"""
This script is used to compute and apply registrations to the voxelwise heatmaps using ANTs.
It is recommended to use this script as the data loader of sade crops the original images,
and therefore there is a mismatch in the image sizes between the heatmaps and the original images.
This script computes registrations on cropped input images using loaders from `sade.datasets`.
Alternatively, one could use their own registration script but should make sure to 
pad the heatmaps to the original image size before applying the registrations.

Example usage:

# This will compute the registrations for the conte dataset
# and save them to the transforms directory specified in the code.
python sade_registration.py --mode compute \
    --config /codespace/sade/sade/configs/ve/biggan_config.py \
    --dataset conte

# This will apply the registrations from the transforms directory
python sade_registration.py --mode apply \
    --config /codespace/sade/sade/configs/ve/biggan_config.py \
    --dataset conte \
    --load_dir /ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/conte \
    --save_dir /ASD/ahsan_projects/Developer/ds-analysis/ebds/registered-heatmaps/

"""
if __name__ == "__main__":
    app.run(main)




""""
NOTE: Check the correct path of braintypicality folder.

Example usage for ABCD-ASD:

# This will compute the registrations for the abcd-asd dataset
# and save them to the transforms directory specified in the code.
python sade_registration_abcd.py --mode compute \
    --config ../sade/sade/configs/ve/biggan_config_abcd_asd.py \
    --dataset abcd-asd

    
# This will apply the registrations from the transforms directory to the original images
python sade_registration_abcd.py --mode apply \
    --config ../sade/sade/configs/ve/biggan_config_abcd_asd.py \
    --dataset abcd-asd \
    --load_dir ../braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/abcd-asd \
    --save_dir ../braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/registered-originals-abcd-asd/ \
    --image_type original

    


# This will apply the registrations from the transforms directory to the heatmaps
python sade_registration_abcd.py --mode apply \
    --config ../sade/sade/configs/ve/biggan_config_abcd_asd.py \
    --dataset abcd-asd \
    --load_dir ../braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/abcd-asd \
    --save_dir ../braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/registered-heatmaps-abcd-asd/ \
    --image_type heatmap



# This will apply the registrations from the transforms directory to the original images (IBIS inlier)
python sade_registration_abcd.py --mode apply \
    --config ../sade/sade/configs/ve/biggan_config_abcd_asd.py \
    --dataset ibis-inlier \
    --load_dir ../braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/ibis-inlier \
    --save_dir ../braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/registered-originals-ibis-inlier/ \
    --image_type original


"""
