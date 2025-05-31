import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import glob
import re
import time
from concurrent.futures import ProcessPoolExecutor
from time import time

import ants
import nibabel as nib
import numpy as np
from tqdm import tqdm

CACHE_DIR = "/ASD/ahsan_projects/braintypicality/dataset/template_cache/"
DATA_DIR = "/BEE/Connectome/ABCD/"

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
crop_idxs_start, crop_idxs_end = 1 + diff // 2, np.array(t1_ref_img.shape) - diff // 2

t1_ref_img = ants.crop_indices(t1_ref_img, crop_idxs_start, crop_idxs_end)
t2_ref_img = ants.crop_indices(t2_ref_img, crop_idxs_start, crop_idxs_end)
ref_img_mask = ants.crop_indices(ref_img_mask, crop_idxs_start, crop_idxs_end)


def extract_brain_mask(
    image, modality="t1", antsxnet_cache_directory=None, verbose=True
):
    from antspynet.utilities import brain_extraction

    if antsxnet_cache_directory is None:
        antsxnet_cache_directory = "ANTsXNet"

    # Truncating intensity as a preprocessing step following the original ants function
    preprocessed_image = ants.image_clone(image)
    truncate_intensity = (0.01, 0.99)
    if truncate_intensity is not None:
        quantiles = (
            image.quantile(truncate_intensity[0]),
            image.quantile(truncate_intensity[1]),
        )
        if verbose:
            print(
                "Preprocessing:  truncate intensities ( low =",
                quantiles[0],
                ", high =",
                quantiles[1],
                ").",
            )

        preprocessed_image[image < quantiles[0]] = quantiles[0]
        preprocessed_image[image > quantiles[1]] = quantiles[1]

    # Brain extraction
    mask = None
    probability_mask = brain_extraction(
        preprocessed_image,
        modality=modality,
        #antsxnet_cache_directory=antsxnet_cache_directory,
        verbose=verbose,
    )
    mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
    mask = ants.morphology(mask, "close", 6).iMath_fill_holes()

    return mask


def register_and_match(
    image,
    target_img=None,
    target_img_mask=None,
    label=None,
    mask=None,
    do_bias_correction=True,
    histogram_match=True,
    histogram_match_points=5,
    compute_brain_mask=True,
    truncate_intensity=(0.01, 0.99),
    modality="t1",
    template_transform_type="Rigid",
    antsxnet_cache_directory=CACHE_DIR,
    verbose=True,
):
    """
    Basic preprocessing pipeline for T1-weighted brain MRI adapted from AntsPyNet
    https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/preprocess_image.py

    Arguments
    ---------
    image : ANTsImage
        input image

    truncate_intensity : 2-length tuple
        Defines the quantile threshold for truncating the image intensity

    modality : string or None
        Modality that defines registration and brain extraction using antspynet tools.
        One of "t1" or "t2"

    template_transform_type : string
        See details in help for ants.registration.  Typically "Rigid" or
        "Affine".
    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary with preprocessing information ANTs image (i.e., source_image) matched to the
    (reference_image).

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> preprocessed_image = preprocess_brain_image(image, do_brain_extraction=False)
    """
    from antspynet.utilities import brain_extraction

    assert modality in ["t1", "t2"]

    preprocessed_image = ants.image_clone(image)

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    # Truncate intensity
    if truncate_intensity is not None:
        quantiles = (
            image.quantile(truncate_intensity[0]),
            image.quantile(truncate_intensity[1]),
        )
        if verbose == True:
            print(
                "Preprocessing:  truncate intensities ( low =",
                quantiles[0],
                ", high =",
                quantiles[1],
                ").",
            )

        preprocessed_image[image < quantiles[0]] = quantiles[0]
        preprocessed_image[image > quantiles[1]] = quantiles[1]

    # Brain extraction
    # mask = None
    if compute_brain_mask:
        probability_mask = brain_extraction(
            preprocessed_image,
            modality=modality,
            #antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose,
        )
        mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
        mask = ants.morphology(mask, "close", 6).iMath_fill_holes()
    # else:
    #     mask = (image > 0).astype("float32")

    # Template normalization
    template_image = t1_ref_img if modality == "t1" else t2_ref_img
    template_img_mask = ref_img_mask

    if target_img is None:
        target_img = template_image
        target_img_mask = ref_img_mask
    # else:
    #     # T1 target img is given, we only need to load T2 Template img
    #     template_image = t2_ref_img

    # Similar to ANTsPyNet we compute the registration via masked images
    target_brain_img = ants.mask_image(target_img, target_img_mask)
    preprocessed_brain_image = ants.mask_image(preprocessed_image, mask)
    registration = ants.registration(
        fixed=target_brain_img,
        moving=preprocessed_brain_image,
        type_of_transform=template_transform_type,
        verbose=verbose,
    )

    # Next we apply the transform to the UNMASKED images
    preprocessed_image = ants.apply_transforms(
        fixed=target_img,
        moving=preprocessed_image,
        transformlist=registration["fwdtransforms"],
        interpolator="linear",
        verbose=verbose,
    )
    mask = ants.apply_transforms(
        fixed=preprocessed_image,
        moving=mask,
        transformlist=registration["fwdtransforms"],
        interpolator="genericLabel",
        verbose=verbose,
    )

    if label:
        label_img = ants.apply_transforms(
            fixed=preprocessed_image,
            moving=label,
            transformlist=registration["fwdtransforms"],
            interpolator="genericLabel",
            verbose=verbose,
        )

    if do_bias_correction:

        # Note that bias correction takes in UNMASKED images
        if verbose:
            print("Preprocessing:  brain correction.")

        n4_output = None
        n4_output = ants.n4_bias_field_correction(
            preprocessed_image,
            mask,
            shrink_factor=4,
            return_bias_field=False,
            verbose=verbose,
        )
        preprocessed_image = n4_output

    preprocessed_image = ants.mask_image(preprocessed_image, mask)

    # Histogram matching with template
    if histogram_match:
        template_brain_image = ants.mask_image(template_image, template_img_mask)
        preprocessed_image = ants.histogram_match_image(
            preprocessed_image,
            template_brain_image,
            number_of_histogram_bins=128,
            number_of_match_points=histogram_match_points,  # Could leave to 1 or 2 if u wanna emphasize intensity during training
        )

    # Min max norm
    preprocessed_image = (preprocessed_image - preprocessed_image.min()) / (
        preprocessed_image.max() - preprocessed_image.min()
    )

    if label:
        return preprocessed_image, label_img

    return preprocessed_image, mask, registration


def preprocessor(sample, save_dir=None):
    # print(SAVE_DIR, TUMOR)
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    subject_id, path = sample

    if DATASET == "TUMOR":
        img = nib.load(path)
        t1_img = ants.from_nibabel(img.slicer[..., 2])
        t2_img = ants.from_nibabel(img.slicer[..., 3])
        _mask = t1_img != 0
    else:
        t1_path = path
        if DATASET == "HCPD":
            t2_path = path.replace("T1w_", "T2w_")
        else:
            t2_path = path.replace("T1w", "T2w")

        t1_img = ants.image_read(t1_path)
        t2_img = ants.image_read(t2_path)

        # t1_mask = extract_brain_mask(
        #     t1_img, antsxnet_cache_directory=CACHE_DIR, verbose=False
        # )
        # t2_mask = extract_brain_mask(
        #     t2_img, antsxnet_cache_directory=CACHE_DIR, verbose=False
        # )

        # Rigid regsiter to MNI + hist normalization + min/max scaling
        t1_img_reg, t1_mask, _ = register_and_match(
            t1_img,
            modality="t1",
            antsxnet_cache_directory=CACHE_DIR,
            verbose=False,
        )

        # Register T2 to the post-registered T1
        t2_img_reg, t2_mask, _ = register_and_match(
            t2_img,
            modality="t2",
            target_img=t1_img_reg,
            target_img_mask=t1_mask,
            antsxnet_cache_directory=CACHE_DIR,
            verbose=False,
        )
    preproc_img = ants.merge_channels([t1_img_reg, t2_img_reg])

    fname = os.path.join(SAVE_DIR, f"{subject_id}.nii.gz")
    preproc_img.to_filename(fname)

    return subject_id


def lesion_preprocessor(sample):
    # print(SAVE_DIR, TUMOR)
    import tensorflow as tf

    # print(f"Using contrast {contrast}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    subject_id, path = sample
    t2_path = path
    t2_img = ants.image_read(t2_path)

    label_path = path.replace("T2w", "T2w_lesion_label")
    # print(label_path)
    label_img = ants.image_read(label_path)

    # if high_contrast:
    t2_img[label_img] = t2_img[label_img] * (contrast / 100)

    # Rigid regsiter to MNI + hist normalization + min/max scaling
    # Additionally register label img to mri
    (
        t2_img,
        label_img,
    ) = register_and_match(
        t2_img,
        label=label_img,
        modality="t2",
        antsxnet_cache_directory=CACHE_DIR,
        verbose=False,
    )

    preproc_img = ants.merge_channels((t2_img, label_img))
    fname = os.path.join(SAVE_DIR, f"{subject_id}.nii.gz")
    preproc_img.to_filename(fname)

    return subject_id


def get_matcher(dataset):

    if dataset == "CONTE-TRIO":
        return re.compile(r"(neo-(\d{4}-\d-\d)|(T\d{4}-\d-\d))")

    if dataset == "TWINS":
        return re.compile(r"(T\d{4}-\d-\d)")

    if dataset == "MSSEG":
        return re.compile(r"UNC_train_(Case\d*)\/")

    if dataset == "CamCAN":
        return re.compile(r"sub-(CC\d*)\/anat")

    if dataset == "MSLUB":
        return re.compile(r"patient(\d\d)_*")
    if dataset == "HCP":
        return re.compile(r"HCP_Data\/(\d*)\/T1w*")

    if dataset == "EBDS":
        return re.compile(r"EBDS/(.*)/\d{1,2}year*")

    if dataset == "HCPD":
        return re.compile(r"(HCD\d*)_V1_MR")

    if dataset == "IBIS":
        return re.compile(r"stx_(\d*)_VSA*_*")

    if dataset == "BraTS-PED":
        return re.compile(r"(BraTS-PED-\d*-\d*)-")

    if dataset == "BraTS-GLI":
        return re.compile(r"(BraTS-GLI-\d*-\d*)-")

    # ABCD adult matcher
    if dataset == "ABCD":
        return re.compile(r"sub-(.*)\/ses-")  # NDAR..?

    matcher = r"neo-(\d{4}-\d-\d)?"

    if dataset == "CONTE2":
        return re.compile(matcher)

    return re.compile("(" + matcher + ")")


def get_abcdpaths(split="train"):

    R = get_matcher("ABCD")

    paths = glob.glob(
        DATA_DIR + "ImageData/Data/*/ses-baselineYear1Arm1/anat/*T1w.nii.gz"
    )
    clean = lambda x: x.strip().replace("_", "")

    file_paths = []

    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)

        if sub_id in inlier_keys:
            inlier_paths.append((sub_id, path))

    print("Collected:", len(inlier_paths))
    return inlier_paths


def get_bratspedpaths(split="train"):
    R = get_matcher("BraTS-PED")
    paths = glob.glob(
        "/ASD2/ahsan_projects/datasets/BRATS2023-PED/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/*/*-t1n.nii.gz"
    )
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))
    return id_paths


def get_bratsgliomapaths(split="train"):
    R = get_matcher("BraTS-GLI")
    paths = glob.glob(
        "/ASD2/ahsan_projects/datasets/BRATS2023-Glioma/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/*/*-t1n.nii.gz"
    )
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))
    print(id_paths[0])
    print("Collected:", len(id_paths))
    return id_paths


def get_ibispaths(split="train"):
    R = get_matcher("IBIS")

    paths = glob.glob(
        "/ASD/Autism/IBIS/Proc_Data/*/VSA*/mri/registered_stx/sMRI/*T1w.nrrd"
    )
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        t2_path = path.replace("T1w", "T2w")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_lesionpaths():
    R = re.compile(r"sub-(.*)_ses-")
    lesion_dir = "/DATA/Users/amahmood/lesions/lesion_load_50_automated/*T2w.nii.gz"
    paths = sorted(glob.glob(lesion_dir))

    id_paths = []
    for path in paths:
        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_hcpdpaths(split="train"):
    R = get_matcher("HCPD")

    paths = glob.glob("/UTexas/HCP/HCPD/fmriresults01/*_V1_MR/T1w/T1w_acpc_dc.nii.gz")
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        t2_path = path.replace("T1w_", "T2w_")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_ebdspaths():
    R = get_matcher("EBDS")
    basepath = "/Human2/ImageImputation/Data/EBDS/"

    paths = glob.glob(f"{basepath}/*/*")
    print("FOUND:", len(list(paths)))

    id_paths = []

    for d in os.listdir(basepath):
        # find school-age chilren
        path = os.path.join(basepath, d)
        ages = [p.split("/")[-1] for p in glob.glob(f"{path}/*")]
        image_path = None

        # 10 year preferred over 8
        for age in ["8year", "10year"]:
            if age in ages:
                path = os.path.join(path, age, "anat")
                t1path = glob.glob(f"{path}/*T1.nrrd")
                t2path = glob.glob(f"{path}/*T2.nrrd")

                if t1path and t2path:
                    image_path = t1path[0]

        if image_path is not None:
            match = R.search(image_path)
            sub_id = match.group(1)
            id_paths.append((sub_id, image_path))

    print("Collected:", len(id_paths))
    return id_paths


def get_contepaths():
    # /conte_projects/CONTE_NEO/Data/neo-0*-10year/Skull_Stripped/neo-0*-10year-T1_SkullStripped_scaled.nrrd

    R = get_matcher("CONTE2")

    basepath = "/Human2/CONTE2/Data/"

    paths = glob.glob(
        f"{basepath}/neo-0*/neo-0*-8year/sMRI/Skull_Stripped/*-T1_Bias_regAtlas_corrected_Stripped_scaled.nrrd"
    )
    paths.extend(
        glob.glob(
            f"{basepath}/neo-0*/neo-0*-10year/sMRI/Skull_Stripped/*-T1_Bias_regAtlas_corrected_Stripped_scaled.nrrd"
        )
    )
    print("FOUND:", len(list(paths)))

    id_paths = {}

    for p in paths:
        t2_path = p.replace("T1_Bias_", "T2_Bias_regT1_")

        # Older way of excluding 8 years was buggy
        # It misssed the 10 years that did not have 8 year scans

        if os.path.exists(t2_path):
            image_path = p
        else:  # No 10 years or 8 years with T1+T2 => skip
            print("Missing T2:", p)
            continue

        match = R.search(image_path)
        sub_id = match.group(1)
        if sub_id in id_paths and "10y" in id_paths[sub_id]:
            continue

        # Some files cause Segmentation faults when read by ANTs :(
        if sub_id in ["0279-1-1", "0311-1-1", "0393-2-1", "0540-1-1"]:
            print("ANTs loading error:", image_path)
            continue
        # _ = ants.image_read(image_path)

        id_paths[sub_id] = image_path
        # id_paths.append((sub_id, image_path))

    id_paths = list(id_paths.items())
    print("Collected:", len(id_paths))
    return id_paths


def get_twinspaths():
    # /Human2/TWINS2/T0*/T0*-8yr/sMRI/Skull_Stripped/T0*-8yr-T1_Bias_regAtlas_corrected_Stripped_scaled.nrrd

    R = get_matcher("TWINS")

    basepath = "/Human2/TWINS2/"

    paths = glob.glob(
        f"{basepath}/T0*/T0*-8yr/sMRI/Skull_Stripped/T0*-8yr-T1_Bias_regAtlas_corrected_Stripped_scaled.nrrd"
    )
    paths.extend(
        glob.glob(
            f"{basepath}/T0*/T0*-10yr/sMRI/Skull_Stripped/T0*-10yr-T1_Bias_regAtlas_corrected_Stripped_scaled.nrrd"
        )
    )
    print("FOUND:", len(list(paths)))

    id_paths = {}

    for p in paths:
        t2_path = p.replace("T1_Bias_", "T2_Bias_regT1_")

        # prefer 10 years
        if os.path.exists(t2_path):
            image_path = p
        else:  # No 10 years or 8 years with T1+T2 => skip
            print("Missing T2:", p)
            continue

        match = R.search(image_path)
        sub_id = match.group(1)
        if sub_id in id_paths and "10y" in id_paths[sub_id]:
            continue

        match = R.search(image_path)
        sub_id = match.group(1)
        id_paths[sub_id] = image_path

    id_paths = list(id_paths.items())
    print("Collected:", len(id_paths))
    return id_paths


def get_contetriopaths():
    # /conte_projects/CONTE_NEO/Data/neo-0*-8year/Skull_Stripped/neo-0*-8year-T1_SkullStripped_scaled.nrrd
    # re.compile(r"(neo-(\d{4}-\d-\d)|(T\d{4}-\d-\d))")
    R = get_matcher("CONTE-TRIO")

    paths = glob.glob(
        "/conte_projects/CONTE_NEO/Data/neo-0*-8year/Skull_Stripped/neo-0*-8year-T1_SkullStripped_scaled.nrrd"
    )
    paths.extend(
        glob.glob(
            "/conte_projects/CONTE_NEO/Data/neo-0*-10year/Skull_Stripped/neo-0*-10year-T1_SkullStripped_scaled.nrrd"
        )
    )
    paths.extend(
        glob.glob(
            "/twin-gilmore/T0*/T0*-8yr/Skull_Stripped/T0*-8yr-T1_SkullStripped_scaled.nrrd"
        )
    )
    paths.extend(
        glob.glob(
            "/twin-gilmore/T0*/T0*-10yr/Skull_Stripped/T0*-10yr-T1_SkullStripped_scaled.nrrd"
        )
    )
    print("FOUND:", len(list(paths)))

    id_paths = {}

    for p in paths:
        t2_path = p.replace("T1_", "T2_")

        if os.path.exists(t2_path):
            image_path = p
        else:  # No 10 years or 8 years with T1+T2 => skip
            print("Missing T2:", p)
            continue

        match = R.search(image_path)
        sub_id = match.group(1)

        # prefer 10 years
        if sub_id in id_paths and "10y" in id_paths[sub_id]:
            continue

        id_paths[sub_id] = image_path

    id_paths = list(id_paths.items())
    print("Collected:", len(id_paths))
    return id_paths


def get_hcppaths(split="train"):
    R = get_matcher("HCP")

    paths = glob.glob("/theia_archive/HCP_Data/*/T1w/T1w_acpc_dc.nii.gz")
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        t2_path = path.replace("T1w_", "T2w_")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_mslubpaths(split="train"):
    R = get_matcher("MSLUB")

    paths = glob.glob(
        "/ASD2/ahsan_projects/datasets/MSLUB/patient*/patient*_T1W.nii.gz"
    )
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        t2_path = path.replace("_T1W", "_T2W")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_camcanpaths(split="train"):
    R = get_matcher("CamCAN")

    paths = glob.glob(
        "/BEE/CamCAN/Download/cc700/mri/pipeline/release004/BIDS_20190411/anat/sub-CC[0-9]*/anat/*_T1w.nii.gz"
    )
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        t2_path = path.replace("_T1w", "_T2w")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))

    return id_paths


def get_mssegpaths(split="train"):
    R = get_matcher("MSSEG")

    paths = glob.glob("/ASD2/ahsan_projects/datasets/NIRAL_MSSeg/*/*_T1.nhdr")
    print("FOUND:", len(list(paths)))

    id_paths = []
    for path in paths:
        t2_path = path.replace("_T1", "_T2")
        if not os.path.exists(t2_path):
            continue

        match = R.search(path)
        sub_id = match.group(1)
        id_paths.append((sub_id, path))

    print("Collected:", len(id_paths))
    return id_paths


def run(paths, process_fn):
    start = time()
    progress_bar = tqdm(
        range(0, len(paths), chunksize),
        total=len(paths) // chunksize,
        initial=0,
        desc="# Processed: ?",
    )

    with ProcessPoolExecutor(max_workers=cpus) as exc:
        for idx in progress_bar:
            idx_ = idx + start_idx
            # print(paths[idx_ : idx_ + chunksize])
            result = list(exc.map(process_fn, paths[idx_ : idx_ + chunksize]))
            progress_bar.set_description("# Processed: {:d}".format(idx_))

    print("Time Taken: {:.3f}".format(time() - start))


# TODO: Add parser to generate each split
if __name__ == "__main__":
    chunksize = cpus = 4
    # cpus = 1
    start_idx = 0

    BASE_DIR = "/DATA/Users/amahmood/braintyp/"
    DATASET = sys.argv[1]
    contrast_experiment = False

    if DATASET == "CONTE-TRIO":
        paths = get_contetriopaths()
    elif DATASET == "TWINS":
        paths = get_twinspaths()
    elif DATASET == "CONTE":
        paths = get_contepaths()
    elif DATASET == "MSSEG":
        paths = get_mssegpaths()
    elif DATASET == "CamCAN":
        paths = get_camcanpaths()
    elif DATASET == "MSLUB":
        paths = get_mslubpaths()
    elif DATASET == "BRATS-GLI":
        paths = get_bratsgliomapaths()
    elif DATASET == "BRATS-PED":
        paths = get_bratspedpaths()
    elif DATASET == "LESION":
        SAVE_DIR = os.path.join(BASE_DIR, "lesion-hc")
        paths = get_lesionpaths()
    # elif DATASET == "TUMOR":
    #     SAVE_DIR = os.path.join(BASE_DIR, "tumor")
    #     paths = get_bratspaths()
    elif DATASET == "IBIS":
        paths = get_ibispaths()
    elif DATASET == "HCPD":
        paths = get_hcpdpaths()
    elif DATASET == "HCP":
        paths = get_hcppaths()
    else:
        paths = get_abcdpaths()

    # if not os.path.exists(SAVE_DIR):
    #     os.makedirs(SAVE_DIR)

    if DATASET == "LESION":
        contrast_multiples = [110, 120, 130, 140]
        process_fn = lesion_preprocessor

        if contrast_experiment:
            for contrast in contrast_multiples:
                print("SAVING HIGH CONTRAST LESIONS!!")
                SAVE_DIR = os.path.join(BASE_DIR, f"lesion-c{contrast}")
                os.makedirs(SAVE_DIR, exist_ok=True)
                run(paths, process_fn)

        else:
            run(paths, process_fn)
