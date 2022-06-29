import argparse
import gc
from natsort import natsorted # pip install natsort
import numpy as np
import os
import nibabel as nib
from glob import glob

# ----------------------------------------------------------------------------
# Argument parser
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
    
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Preprocessing a folder with .nii files")
    # data specs
    parser.add_argument("--path_folder_with_raw_data", type=str, default= "F:/MA/lits2017_data/full_dataset/raw",
                        # default="F:/MA/ircad-dataset/raw/train",
                        help="")
    parser.add_argument("--path_folder_processed_data", type=str, default= "F:/MA/lits2017_data/full_dataset/processed",
                        # default="F:/MA/ircad-dataset/preprocessed/train",
                        help="")
    parser.add_argument("--substr_image", type=str, default = "volum",
                        # default="_orig",
                        help="substring to identify source image file")
    parser.add_argument("--substr_segm", type=str, default = "segmentat",
                        # default="_liver",
                        help="substring to identify source masking/segmentation file")
    parser.add_argument("--suffix", type=str, default = "-",
                        # default="e",
                        help="delimiting suffix to check numbering")
    parser.add_argument("--file_name_processed_images", type=str, default="imgs_train.npy",
                        help="save as this file name")
    parser.add_argument("--file_name_processed_masks", type=str, default="masks_train.npy",
                        help="save as this file name")
    parser.add_argument("--training_files", type=bool_flag, default="true",
                        help="are we processing files for training or for validation?")
    parser.add_argument("--image_rows", type=int, default=int(512/2),
                        help="# convert to this image size (width)"),
    parser.add_argument("--image_cols", type=int, default=int(512/2),
                        help="convert to this image size (height)"),
    parser.add_argument("--skip_every", type=int, default=4,
                    help="undersample every skp_every-th z axis (axial) slice"),
    # training options
    parser.add_argument("--verbose_terminal", type=bool_flag, default="true",
                        help="how many in-between infos to print when executing this script. True = print all; False = print little")
    
    
    
    return parser

# ----------------------------------------------------------------------------
# Helper functions
def get_clean_path(path: str):
    return os.path.normpath(path.replace("\\","/").replace("\r", "/r").replace("\n", "/n"))

def get_substrings(path_to_folder: list[str], str_match: str = '_liver') -> list[str]:
    """
    In a folder, identify all files which include the substring str_match.
    
    Args:
        path_to_folder (list[str]): list of strings pointing to a paths.
        str_match (str): substring to look for in path_to_folder.
    Returns:
        List of strings of file names that match str_match
    """
    list_with_matched_substrings = list(filter(lambda x: str_match in x, path_to_folder))
    return list_with_matched_substrings

def get_distinct_substrings(string):
    j=1
    a=set()
    while True:
        for i in range(len(string)-j+1):
            a.add(string[i:i+j])
        if j==len(string):
            break
        j+=1
    return a
# ----------------------------------------------------------------------------
# Main preprocesing function
def preprocess(path_folder_with_raw_data: str, 
               path_folder_processed_data: str, 
               substr_image: str = "_orig",
               substr_mask: str = "_liver",
               suffix: str = "e",
               file_name_processed_images: str = 'imgs_train.npy',
               file_name_processed_masks: str = 'masks_train.npy',
               training_files: bool = True,
               skip_every: int = 2,
               verbose_terminal: bool = False) -> None:
    """
    Process entire folders filled with niftis. put out numpy array that can be loaded for training
    
    Args:
        skip_every (int) : how many slices to skip i.o.t. downsample.
    Returns: 
        None. But saves 2 .npy files in specified locations (the folder w/ processed data)
    """
    image_rows: int = int(512/skip_every)
    image_cols: int = int(512/skip_every)
    nifti_files = natsorted(glob(os.path.join(path_folder_with_raw_data, '*')))
    # file names corresponding to training masks
    segm_masks = get_substrings(nifti_files, substr_mask)
    # file names corresponding to training images
    source_images = get_substrings(nifti_files, substr_image)
    assert ((len(segm_masks) is not int(0)) and (len(source_images) is not int(0))), "check path and substrings to search for"
    # --------------------------------------------------------------
    # double check that numbering matches

    # "e01", "e02" in file names
    # numbers_range = range(1, 10)
    # sequence_of_numbers = [number for number in numbers_range]
    # idx = ["0"] + [suffix + str(number) for number in sequence_of_numbers]

    # numbers_range = range(10,21)
    # sequence_of_numbers = [number for number in numbers_range]
    # idx = idx + [suffix + str(number) for number in sequence_of_numbers]

    # counter = int(0)
    # for liver, orig in zip(segm_masks, source_images):
    #     if not (idx[counter] in liver or idx[counter] in orig):
    #         raise AssertionError("indices mismatch!!!!")
    #     counter += 1
    # if params.verbose_terminal:
    #     print("seems that the indices all match as they should")
    # # --------------------------------------------------------------
    masks_list = []
    images_list = []

    if training_files:
        for liver_mask, orig_scan in zip(segm_masks, source_images):
            # load 3D training segmentation mask (shape=(512,512,129))
            if verbose_terminal:
                print(liver_mask)
            mask_nifti = nib.load(os.path.join(path_folder_with_raw_data, liver_mask))
            # load 3D training ground truth image
            # if verbose_terminal:
            #     print(orig_scan)
            image_nifti = nib.load(os.path.join(path_folder_with_raw_data, orig_scan)) 
            if verbose_terminal:
                print(f"Processing {orig_scan}")
            for k in range(mask_nifti.shape[2]-1):
                # axial cuts are made along the z axis with undersampling ->  downsample each slice by omitting every skip_every-th row/col
                # create np.arrays with numpy short unsigned integer. Makes a huge difference i.t.o. memory.
                mask_2d = np.array(mask_nifti.get_fdata()[::skip_every, ::skip_every, k], dtype = np.uint8)  
                # create np.arrays with numpy short unsigned integers.
                # This naturally crops the values to max. 255 which is fine since liver tissue 
                image_2d = np.array(image_nifti.get_fdata()[::skip_every, ::skip_every, k], dtype = np.int16)
                # only recover the 2D sections containing the liver
                # if mask_2d contains only 0, it means that there is no liver
                if len(np.unique(mask_2d)) != 1:
                    # print(k)
                    masks_list.append(mask_2d)
                    images_list.append(image_2d)
    else:
        for liver_mask, orig_scan in zip(segm_masks, source_images):
            mask_nifti = nib.load(os.path.join(path_folder_with_raw_data, liver_mask))
            # load 3D training ground truth image
            image_nifti = nib.load(os.path.join(path_folder_with_raw_data, orig_scan)) 
            assert mask_nifti.shape[2] == image_nifti.shape[2]
            
            for k in range(mask_nifti.shape[2]):  
                masks_list.append(np.array(mask_nifti.get_fdata()[::2, ::2, k]))
                images_list.append(np.array(image_nifti.get_fdata()[::2, ::2, k]))
                    
    imgs = np.ndarray((len(images_list), image_2d.shape[0], image_2d.shape[1]),
                      dtype='int16')
    for index, img in enumerate(images_list):
        imgs[index, :, :] = img
    np.save(get_clean_path(os.path.join(path_folder_processed_data, file_name_processed_images)), imgs)
    del imgs
    del images_list
    gc.collect()
    
    imgs_mask = np.ndarray((len(masks_list), mask_2d.shape[0], mask_2d.shape[1]), 
                           dtype='uint8')
    for index, img in enumerate(masks_list):
        imgs_mask[index, :, :] = img
    np.save(get_clean_path(os.path.join(path_folder_processed_data, file_name_processed_masks)), imgs_mask)
    del imgs_mask
    del masks_list
    gc.collect()
    
    print('Images and masks saved to .npy files at:',
          '\n',
          os.path.join(path_folder_processed_data, file_name_processed_images),
          '\n', 
          os.path.join(path_folder_processed_data, file_name_processed_masks)
          )

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # generate parser / parse parameters
    params = get_parser().parse_args()
        
    preprocess(path_folder_with_raw_data = params.path_folder_with_raw_data,
               path_folder_processed_data = params.path_folder_processed_data,
               substr_image = params.substr_image,
               substr_mask = params.substr_segm,
               suffix = params.suffix,
               file_name_processed_images = params.file_name_processed_images,
               file_name_processed_masks = params.file_name_processed_masks,
               training_files = params.training_files,
               skip_every= params.skip_every,
               verbose_terminal = params.verbose_terminal)
    print("preprocessing complete.")