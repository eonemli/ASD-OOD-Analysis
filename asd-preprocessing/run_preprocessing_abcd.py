import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from time import time

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from mri_utils import (
    get_bratsgliomapaths,
    get_bratspedpaths,
    get_camcanpaths,
    get_contepaths,
    get_ebdspaths,
    get_hcpdpaths,
    get_hcppaths,
    get_ibispaths,
    get_mslubpaths,
    get_mssegpaths,
    get_twinspaths,
    get_contetriopaths,
    register_and_match,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


DATADIR = "/BEE/Connectome/ABCD/"
# CACHEDIR = "/ASD/ahsan_projects/braintypicality/dataset/template_cache/"
CACHEDIR = "../braintypicality/dataset/template_cache"

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("Found GPUs:", gpus)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def runner(
    path,
    dataset,
    save_sub_dir="processed_v3",
    label_img=False,
    run_segmentation=False,
    compute_brain_mask=True,
    do_bias_correction=True,
    histogram_match=True,
    histogram_match_points=5,
):
    import ants
    import antspynet
    from antspynet.utilities import preprocess_brain_image

    t1_path = t2_path = label_path = None
    os.makedirs(f"/{DATADIR}/Users/emre/braintyp/{save_sub_dir}", exist_ok=True)

    if dataset == "ABCD":
        R = re.compile(r"Data_abcd_asd_scr_pos_gz\/sub-(.*)\/ses-")
        subject_id = R.search(path).group(1)
        t1_path = path
        t2_path = path.replace("T1w", "T2w")
    elif dataset == "IBIS":
        subject_id, t1_path = path
        subject_id = dataset + subject_id
        t2_path = t1_path.replace("T1w", "T2w")
    elif dataset == "EBDS":
        subject_id, t1_path = path
        subject_id = dataset + subject_id
        t2_path = t1_path.replace("T1.nrrd", "T2.nrrd")
    elif dataset == "HCPD":
        subject_id, t1_path = path
        t2_path = t1_path.replace("T1w_", "T2w_")
    elif dataset in ["BRATS-PED", "BRATS-GLI"]:
        subject_id, t1_path = path
        t2_path = t1_path.replace("t1n.nii.gz", "t2w.nii.gz")
        label_path = t1_path.replace("t1n.nii.gz", "seg.nii.gz")
    elif dataset == "HCP":
        subject_id, t1_path = path
        t2_path = t1_path.replace("T1w_", "T2w_")
        subject_id = dataset + subject_id
    elif dataset == "MSLUB":
        subject_id, t1_path = path
        t2_path = t1_path.replace("T1W.nii.gz", "T2W.nii.gz")
        dirpath = os.path.dirname(t1_path).replace("/raw", "")
        # sid = os.path.basename(dirpath)
        # label_path = os.path.join(dirpath, f"{sid}_consensus_gt.nii.gz")
        label_path = t1_path.replace("T1W.nii.gz", "consensus_gt.nii.gz")
        mask_path = label_path.replace("consensus_gt.nii.gz", "brainmask.nii.gz")
        subject_id = dataset + subject_id
    elif dataset == "CAMCAN":
        subject_id, t1_path = path
        t2_path = t1_path.replace("_T1w", "_T2w")
        subject_id = dataset + subject_id
    elif dataset == "MSSEG":
        subject_id, t1_path = path
        t2_path = t1_path.replace("_T1", "_T2")
        subject_id = dataset + subject_id
        label_path = t1_path.replace("_T1", "_lesion")
    elif dataset in ["CONTE", "TWINS"]:
        subject_id, t1_path = path
        t2_path = t1_path.replace("T1_Bias_", "T2_Bias_regT1_")
        subject_id = dataset + subject_id
    elif dataset == "CONTE-TRIO":
        subject_id, t1_path = path
        t2_path = t1_path.replace("T1_", "T2_")
        subject_id = dataset + subject_id
    else:
        raise NotImplementedError

    t1_img = ants.image_read(t1_path)
    t2_img = ants.image_read(t2_path)

    mask_img = None

    if dataset in ["CONTE-TRIO", "TWINS", "CONTE", "BRATS-PED", "BRATS-GLI", "EBDS"]:
        assert not compute_brain_mask, "{dataset} is already brain masked"
        mask_img = (t1_img > 0).astype("float32")

    # if dataset == "MSLUB":
    #     assert not compute_brain_mask, "MSLUB has precomputed brain masks"
    #     mask_img = ants.image_read(mask_path)
    #     t1_img *= mask_img
    #     t2_img *= mask_img

    # Rigid regsiter to MNI + hist normalization + min/max scaling
    t1_img, t1_mask, registration = register_and_match(
        t1_img,
        modality="t1",
        antsxnet_cache_directory=CACHEDIR,
        verbose=False,
        mask=None if compute_brain_mask else mask_img,
        compute_brain_mask=compute_brain_mask,
        histogram_match=histogram_match,
        do_bias_correction=do_bias_correction,
        histogram_match_points=histogram_match_points,
    )

    # Register t2 to the t1 already registered to MNI above
    t2_img, t2_mask, _ = register_and_match(
        t2_img,
        modality="t2",
        target_img=t1_img,
        target_img_mask=t1_mask,
        antsxnet_cache_directory=CACHEDIR,
        verbose=False,
        mask=None if compute_brain_mask else mask_img,
        compute_brain_mask=compute_brain_mask,
        histogram_match=histogram_match,
        do_bias_correction=do_bias_correction,
        histogram_match_points=histogram_match_points,
    )

    # Further apply the opposite modality masks to get a tighter brain crop
    # the same as t1_mask & t2_mask
    if compute_brain_mask:
        combined_mask = t1_mask * t2_mask
        t1_img = ants.mask_image(t1_img, combined_mask)
        t2_img = ants.mask_image(t2_img, combined_mask)

    preproc_img = ants.merge_channels([t1_img, t2_img])
    fname = os.path.join(
        f"/{DATADIR}/Users/emre/braintyp/{save_sub_dir}", f"{subject_id}.nii.gz"
    )
    preproc_img.to_filename(fname)

    # Register label segmentations to new t1
    if label_img:
        assert label_path is not None
        lab_img = ants.image_read(label_path)
        lab_img = ants.apply_transforms(
            fixed=t1_img,
            moving=lab_img,
            transformlist=registration["fwdtransforms"],
            interpolator="genericLabel",
        )
        fname = os.path.join(
            f"/{DATADIR}/Users/emre/braintyp/{save_sub_dir}",
            f"{subject_id}_label.nii.gz",
        )
        lab_img.to_filename(fname)

    if run_segmentation:
        t1_preprocessing = preprocess_brain_image(
            t1_img,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality=None,
            template="croppedMni152",
            template_transform_type="antsRegistrationSyNQuickRepro[a]",
            do_bias_correction=True,
            do_denoising=True,
            verbose=False,
        )
        t1_preprocessed = t1_preprocessing["preprocessed_image"]

        t1_seg = antspynet.utilities.deep_atropos(
            t1_preprocessed,
            do_preprocessing=False,
            #             antsxnet_cache_directory=CACHEDIR,
            verbose=True,
        )["segmentation_image"]

        t1_seg.to_filename(
            f"/{DATADIR}/Users/emre/braintyp/segs/{subject_id}.nii.gz"
        )

        # Also register segmentations to new t1
        t1_seg = ants.apply_transforms(
            fixed=t1_img,
            moving=t1_seg,
            transformlist=registration["fwdtransforms"],
            interpolator="genericLabel",
        )

        wm_mask = t1_seg == 3
        t1_wm = t1_img * wm_mask
        t1_wm = t1_wm[t1_wm > 0].ravel()

        t2_wm = t2_img * wm_mask
        t2_wm = t2_wm[t2_wm > 0].ravel()

        # Save outputs
        fname = os.path.join(
            f"/{DATADIR}/Users/emre/braintyp/segs/", f"{subject_id}.npz"
        )
        np.savez_compressed(fname, **{"t1": t1_wm, "t2": t2_wm})

    return


def run(paths, process_fn, chunksize=1):
    start_idx = 0
    start = time()
    progress_bar = tqdm(
        range(0, len(paths), chunksize),
        total=len(paths) // chunksize,
        initial=0,
        desc="# Processed: ?",
    )

    # with ProcessPoolExecutor(max_workers=chunksize) as exc:
    #     for idx in progress_bar:
    #         idx_ = idx + start_idx
    #         result = list(exc.map(process_fn, paths[idx_ : idx_ + chunksize]))
    #         progress_bar.set_description("# Processed: {:d}".format(idx_))

    for idx in progress_bar:
        idx_ = idx + start_idx
        if idx_ > len(progress_bar):
            break
        process_fn(paths[idx_])
        progress_bar.set_description("# Processed: {:d}".format(idx_))

    print("Time Taken: {:.3f}".format(time() - start))


if __name__ == "__main__":
    dataset = sys.argv[1]
    assert dataset in [
        "CONTE-TRIO",
        "TWINS",
        "CONTE",
        "MSSEG",
        "CAMCAN",
        "MSLUB",
        "HCP",
        "BRATS-GLI",
        "BRATS-PED",
        "EBDS",
        "IBIS",
        "HCPD",
        "ABCD",
    ], "Dataset name must be defined"

    if dataset == "CONTE-TRIO":
        file_paths = get_contetriopaths()
        run(
            file_paths,
            partial(
                runner,
                dataset=dataset,
                save_sub_dir=dataset.lower(),
                run_segmentation=False,
                compute_brain_mask=False,
            ),
        )
    elif dataset == "TWINS":
        file_paths = get_twinspaths()
        run(
            file_paths,
            partial(
                runner,
                dataset=dataset,
                save_sub_dir=dataset.lower(),
                run_segmentation=False,
                compute_brain_mask=False,
            ),
        )
    elif dataset == "CONTE":
        file_paths = get_contepaths()
        run(
            file_paths,
            partial(
                runner,
                dataset=dataset,
                save_sub_dir=dataset.lower(),
                run_segmentation=False,
                compute_brain_mask=False,
            ),
        )
    elif dataset == "MSSEG":
        file_paths = get_mssegpaths()
        run(
            file_paths,
            partial(
                runner,
                dataset=dataset,
                save_sub_dir=dataset.lower(),
                run_segmentation=False,
                label_img=True,
            ),
        )
    elif dataset == "CAMCAN":
        file_paths = get_camcanpaths()
        run(
            file_paths,
            partial(
                runner,
                dataset=dataset,
                save_sub_dir=dataset.lower(),
                run_segmentation=False,
            ),
        )
    elif dataset in ["BRATS-PED", "BRATS-GLI"]:
        if dataset == "BRATS-PED":
            file_paths = get_bratspedpaths()
        else:
            file_paths = get_bratsgliomapaths()
        run(
            file_paths,
            partial(
                runner,
                dataset=dataset,
                save_sub_dir=dataset.lower() + "-hist-m2",
                compute_brain_mask=False,
                label_img=True,
                run_segmentation=False,
                histogram_match_points=2,
            ),
        )
    elif dataset == "MSLUB":
        file_paths = get_mslubpaths()
        run(
            file_paths,
            partial(
                runner,
                dataset=dataset,
                save_sub_dir=dataset.lower() + "-hist-m2",
                label_img=True,
                run_segmentation=False,
                compute_brain_mask=True,
                do_bias_correction=True,
                histogram_match=True,
                histogram_match_points=2,
            ),
        )
    elif dataset == "HCP":
        file_paths = get_hcppaths()
        run(file_paths, partial(runner, dataset=dataset))
    elif dataset == "EBDS":
        file_paths = get_ebdspaths()
        run(
            file_paths,
            partial(runner, dataset=dataset, compute_brain_mask=False),
        )
    elif dataset == "IBIS":
        file_paths = get_ibispaths()
        run(file_paths, partial(runner, dataset=dataset))
    elif dataset == "HCPD":
        file_paths = get_hcpdpaths()
        run(file_paths, partial(runner, dataset=dataset))
    else:  # get abcd paths
        paths = glob.glob(
            f"/{DATADIR}/ImageData/Data_abcd_asd_scr_pos_gz/*/ses-baselineYear1Arm1/anat/*T1w.nii.gz"
        )
        R = re.compile(r"Data_abcd_asd_scr_pos_gz\/sub-(.*)\/ses-")
        clean = lambda x: x.strip().replace("_", "")

        with open("abcd_qc_passing_keys_160.txt", "r") as f:
            abcd_qc_keys = set([clean(x) for x in f.readlines()])

        file_paths = []
        id_checker = lambda x: R.search(x).group(1) in abcd_qc_keys
        file_paths = list(filter(id_checker, paths))
        #assert len(file_paths) == len(abcd_qc_keys)

        print(f"Total paths found: {len(paths)}")
        print(f"Subject IDs loaded: {len(abcd_qc_keys)}")

        file_paths = list(filter(id_checker, paths))
        print(f"Filtered paths after matching: {len(file_paths)}")

        # Optionally: print a few examples
        print("Example path:", file_paths[0] if file_paths else "No paths found")



        #run(file_paths, runner)
        run(file_paths, partial(runner, dataset=dataset))
