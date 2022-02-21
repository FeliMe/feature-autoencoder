import argparse
from glob import glob
import os
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tqdm import tqdm

from fae import DATAROOT
from fae.utils.registrators import MRIRegistrator
from fae.utils.robex import strip_skull_ROBEX


class BraTSHandler():
    def __init__(self, args):
        """ipp.cbica.upenn.edu 2020 version"""
        self.prepare_BraTS(args)
        self.registerBraTS(args)

    def prepare_BraTS(self, args):
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'BraTS')

        if not os.path.exists(os.path.join(args.dataset_path, 'MICCAI_BraTS2020_TrainingData.zip')):
            raise RuntimeError(f"Apply for the BraTS2020 data at ipp.cbica.upenn.edu"
                               f" and download it to {args.dataset_path}")

        self.unzip_BraTS(
            dataset_path=args.dataset_path,
            force=False
        )
        self.rename_lesions(args)

    @staticmethod
    def unzip_BraTS(dataset_path, force=False):
        train_zip = os.path.join(
            dataset_path, 'MICCAI_BraTS2020_TrainingData.zip')
        val_zip = os.path.join(
            dataset_path, 'MICCAI_BraTS2020_ValidationData.zip')

        train_dir = os.path.join(dataset_path, 'MICCAI_BraTS2020_TrainingData')
        val_dir = os.path.join(dataset_path, 'MICCAI_BraTS2020_ValidationData')

        # Remove target directories if force
        if force:
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(val_dir, ignore_errors=True)

        # Extract zip
        print(f"Extracting {train_zip}")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print(f"Extracting {val_zip}")
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)

    @staticmethod
    def rename_lesions(args):
        print("Renaming segmentation files in BraTS to "
              "'anomaly_segmentation_unregistered.nii.gz'")
        lesion_files = glob(f"{args.dataset_path}/*/*/*_seg.nii.gz")
        target_files = [
            '/'.join(f.split('/')[:-1] + ['anomaly_segmentation_unregistered.nii.gz']) for f in lesion_files]
        for lesion, target in zip(lesion_files, target_files):
            data = nib.load(lesion, keep_file_open=False)
            volume = data.get_fdata(caching='unchanged',
                                    dtype=np.float32).astype(np.dtype("short"))
            nib.save(nib.Nifti1Image(volume, data.affine), target)
            # shutil.copy(lesion, target)

    @staticmethod
    def registerBraTS(args):
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'BraTS')

        print("Registering BraTS")

        # Get all files
        files = glob(
            f"{args.dataset_path}/MICCAI_BraTS2020_TrainingData/*/*_t1.nii.gz")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        # template_path = os.path.join(
        #     DATAROOT, f'BrainAtlases/mni_icbm152_nlin_sym_09a/{w.lower()}_stripped.nii')
        template_path = os.path.join(
            DATAROOT, 'BrainAtlases/sri24_spm8/templates/T1_brain.nii')
        # registrator = SitkRegistrator(template_path)
        registrator = MRIRegistrator(template_path=template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path[:path.rfind("t1")]
            folder = '/'.join(path.split('/')[:-1])
            # Transform T2 image
            path = base + "t2.nii.gz"
            save_path = base + "t2_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )
            # Transform FLAIR image
            path = base + "flair.nii.gz"
            save_path = base + "flair_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )
            # Transform segmentation
            path = os.path.join(
                folder, "anomaly_segmentation_unregistered.nii.gz")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )


class CamCANHandler():
    def __init__(self, args):
        """https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/index.php"""
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'CamCAN')

        print("Preparing CamCAN directory")
        self.prepare_CamCAN(args)
        print(f"Skull stripping for CamCAN {args.weighting} scans")
        self.skull_strip_CamCAN(args)
        self.register_CamCAN(args)

    @staticmethod
    def prepare_CamCAN(args):
        # Check if data is downloaded
        if not os.path.exists(os.path.join(args.dataset_path, 'cc700')):
            raise RuntimeError("Missing dataset. Apply for CamCAN data and "
                               f"place it into {args.dataset_path}")

        # Move the data to a 'normal' directory
        normal_dir = os.path.join(args.dataset_path, 'normal')
        os.makedirs(normal_dir, exist_ok=True)

        patient_dirs = glob(
            f"{os.path.join(args.dataset_path, 'cc700/mri/pipeline/release004/BIDS_20190411/anat')}/sub*/")
        for d in tqdm(patient_dirs):
            # Move all files from 'anat' dir to parent dir
            for f in glob(f"{d}anat/*"):
                shutil.move(f, d)

            # Remove the empty 'anat' directory
            shutil.rmtree(f"{d}anat/", ignore_errors=True)

            # Move the directory
            shutil.move(d, normal_dir)

    @staticmethod
    def register_CamCAN(args):

        print("Registering CamCAN")

        # Get all files
        files = glob(
            f"{os.path.join(args.dataset_path, 'normal')}/*/*T1w_stripped.nii.gz")
        print(f"Found {len(files)} files")

        if len(files) == 0:
            raise RuntimeError("Found 0 files")

        # Initialize the registrator
        template_path = os.path.join(
            DATAROOT, 'BrainAtlases/sri24_spm8/templates/T1_brain.nii')
        registrator = MRIRegistrator(template_path=template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path[:path.rfind("T1")]
            # Transform T2 image
            path = base + "T2w_stripped.nii.gz"
            save_path = base + "T2w_stripped_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )

    def skull_strip_CamCAN(self, args):
        w = args.weighting
        if not isinstance(w, str):
            raise RuntimeError(f"Invalid value for --weighting {w}")
        # Get list of all files
        paths = glob(
            f"{os.path.join(args.dataset_path, 'normal')}/*/*{w.upper()}w.nii.gz")
        print(f"Found {len(paths)}")

        if len(paths) == 0:
            raise RuntimeError("No paths found")

        if w.lower() == 't1':
            # Run ROBEX
            strip_skull_ROBEX(paths)
        elif w.lower() == "t2":
            self.skull_strip_CamCAN_T2(paths)
        else:
            raise NotImplementedError("CamCAN skull stripping not implemented"
                                      f" for --weighting {w}")

    @staticmethod
    def skull_strip_CamCAN_T2(paths):
        """Skull strip the registered CamCAN T2 images with the results
        of the skull stripped registered CamCAN T1 images
        """
        for path in tqdm(paths):
            t1_stripped_path = f"{path[:path.rfind('T2w')]}T1w_stripped.nii.gz"
            t2_stripped_path = f"{path[:path.rfind('T2w')]}T2w_stripped.nii.gz"
            if not os.path.exists(t1_stripped_path):
                print(f"WARNING: No T1 skull stripped file found for {path}")
            # Load T2 weighted scan
            t2_data = nib.load(path)
            affine = t2_data.affine
            t2 = np.asarray(t2_data.dataobj, dtype=np.short)
            # Load T1 skull stripped scan
            t1_stripped = np.asarray(
                nib.load(t1_stripped_path).dataobj, dtype=np.short)
            t2_stripped = t2.copy()
            t2_stripped[t1_stripped == 0] = 0
            # Save skull stripped t2
            nib.save(nib.Nifti1Image(t2_stripped.astype(
                np.short), affine), t2_stripped_path)


def prepare_data(args):
    if args.dataset == 'CamCAN':
        CamCANHandler(args)
    elif args.dataset == 'BraTS':
        BraTSHandler(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CamCAN', 'BraTS'])
    parser.add_argument('--force_download', action='store_true')
    parser.add_argument('--force_split', action='store_true')
    parser.add_argument('--dataset_path', default=None)
    # Preprocessing arguments
    parser.add_argument('--weighting', type=str,
                        choices=['t1', 't2', 'T1', 'T2', 'FLAIR'])
    args = parser.parse_args()

    # Add this to handle ~ in path variables
    if args.dataset_path:
        args.dataset_path = os.path.expanduser(args.dataset_path)

    prepare_data(args)
