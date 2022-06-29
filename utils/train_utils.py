import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset

from utils.swag_custom import SWAG
from u_net.unet import UNet

def get_pixels_hu(scans):
    """
    Convert raw values into Hounsfield units
    """
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# ----------------------------------------------------------------------------
# Argument parser
def _bool_flag(s):
    """
    Parse boolean arguments from command line.
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
    parser = argparse.ArgumentParser(description="Training UNet with preprocessed files")
    # data paths
    parser.add_argument("--path_train_images", type=str, 
                        # default='F:/MA/ircad-dataset/preprocessed/train/imgs_train.npy',
                        default = "F:/MA/lits2017_data/small_dataset/processed/imgs_train.npy",
                        help="")
    parser.add_argument("--path_train_masks", type=str, 
                        # default='F:/MA/ircad-dataset/preprocessed/train/masks_train.npy',
                        default = "F:/MA/lits2017_data/small_dataset/processed/masks_train.npy",
                        help="")
    parser.add_argument("--path_eval_images", type=str,
                        # default='F:/MA/ircad-dataset/preprocessed/eval/imgs_eval.npy',
                        # default = "F:/MA/lits2017_data/small_dataset/processed/",
                        help="")
    parser.add_argument("--path_eval_masks", type=str,
                        # default='F:/MA/ircad-dataset/preprocessed/eval/masks_eval.npy',
                        # default = "F:/MA/lits2017_data/small_dataset/processed/",
                        help="")  
    parser.add_argument("--path_model", type=str,
                        # default='F:/MA/ircad-dataset/preprocessed/eval/masks_eval.npy',
                        default = "F:/MA/lits2017_data/model/",
                        help="")  
    # CUDA stuff
    parser.add_argument("--use_cuda", type = _bool_flag, default="true", help="")
    parser.add_argument("--amp", type = _bool_flag, default="true", help="torch.cuda mixed precision")
    # "tunable" hyperparams
    parser.add_argument("--optm", type=str, default='SGD', 
                        choices=["SGD", "Adam", "RMSProp"], 
                        help="which optimizer")
    parser.add_argument("--sgd_momentum", type=float, default=0, 
                        help= "momentum for SGD")
    parser.add_argument("--adam_betas", type=float, default=[0.5, 0.999], nargs=2, 
                        help="coefficients used for computing running averages of gradient and its square")    
    parser.add_argument("--criterion", type=str, 
                        default='soft_dice_coeff', 
                        # default='cross_entropy', 
                        choices=["cross_entropy", "soft_dice_coeff"],
                        help="which loss function to use")
    parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="")
    parser.add_argument("--tolerance", type=int, default=10, help="")
    # dataloader
    parser.add_argument("--take_small_subset", type=_bool_flag, default="true",
                        help="do training on n=100 subset")
    parser.add_argument("--val_percent", type=float, default=0.2, help="percentage for eval set (not test set!)")
    parser.add_argument("--num_workers", type=int, default=4, help="distribute batches into subprocesses (?)")
    parser.add_argument("--batch_size", type=int, default=6, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    # Net
    parser.add_argument("--net", type=str, default='UNet', help="which architecture")
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--n_channels', type=int, default=1, help='Number of channels (grayscale => 1)')
    # SWAG specific
    parser.add_argument("--burn_in", type=int, default=1, help="")
    parser.add_argument('--max_num_models', type=int, default=0, help='')
    parser.add_argument('--mc_samples', type=int, default=3, help='')
    parser.add_argument('--no_cov_mat', type=_bool_flag, default="True", help='')
    parser.add_argument('--var_clamp', type=float, default=1e-30, help='')
    # meta info, tracking
    parser.add_argument("--verbose_terminal", type=_bool_flag, default="true",
                    help="how many in-between infos to print when executing this script. True = print all; False = print little")
    parser.add_argument("--epochs_to_check_val_at", type=int, default=1, 
                        help="evaluate performance on validation set every couple epochs.")
    parser.add_argument("--wandb", type=_bool_flag, default="false",
                        help="enable online tracking with WandB. Otherwise offline tracking with custom Tracker class.")
    return parser

def assert_parser_params(parser) -> None:
    """
    Make relevant assertion checks for every parameter in the argparse parser.
    Args:
        parser (argparse.ArgumentParser.parse_args()): parsed arguments
    Returns:
        None
    """
    # TODO maybe put this in a pytest kind of file
    assert (parser.criterion == "cross_entropy") or (parser.criterion =='soft_dice_coeff')
    assert os.path.exists(parser.path_train_images)
    assert os.path.exists(parser.path_train_masks)
    # assert os.path.exists(parser.path_eval_images)
    # assert os.path.exists(parser.path_eval_masks)
    assert parser.burn_in < parser.epochs, "please ensure burn_in < epochs"
    assert parser.burn_in >= parser.epochs_to_check_val_at, "please set epochs_to_check_val_at >= burn_in"
    # and so on

# ----------------------------------------------------------------------------
# Helper functions
def load_processed_data(file_processed_images: str = None,
                        file_processed_masks: str = None):
    imgs = np.load(os.path.join(file_processed_images))
    masks = np.load(os.path.join(file_processed_masks))
    # TODO assert imgs + masks type and that it is not empty
    # TODO maybe add type hint to function
    return imgs, masks

# make dataloader. depending on loss function, data must be encoded differently
def make_dataloaders(imgs: np.ndarray,
                     masks: np.ndarray,
                     loader_args: dict,        
                     criterion: str,
                     val_percent: float = 0.2,
                     seed = 42
                     ):
    """
    Make torch DataLoaders, with suiting dimensions depending on criterion.
    Dataloader should already have the data in the fitting dimension+format. This is 
    better than to fiddle with dimensions right when the loss is calculated.
    # TODO might be helpful to use ImageDataset class instead since it allows for easier augmentation
    Take images and masks and return 2 DataLoaders: one for training, one for validation
    
    Args:
        imgs: np.ndarray
        masks: np.ndarray
        loader_args: dict   
        criterion: str
        val_percent: float = 0.2
    
    Return: train_loader = torch.utils.DataLoader, val_loader = torch.utils.DataLoader, n_train, n_val
    """
    # Split into train / validation partitions
    n_val = int(len(imgs) * val_percent)
    n_train = len(imgs) - n_val
    assert (n_train + n_val) == len(imgs), "rounding error, omitted training data"
    
    # ------------------
    # unsqueeze(0).permute(..) stuff: add channel depth = 1 so that the dataloader doesnt mess up the dimensions
    imgs = torch.as_tensor(imgs).unsqueeze(0).permute(1,0,2,3).float()
    
    if criterion == "cross_entropy":
        # masks = torch.as_tensor(masks).unsqueeze(0).permute(1,0,2,3).float()
        masks = torch.nn.functional.one_hot(torch.as_tensor(masks).long()).permute(0,3,1,2).float()
    elif criterion =="soft_dice_coeff":
        # one hot encoding
        masks = torch.nn.functional.one_hot(torch.as_tensor(masks).long()).permute(0,3,1,2).float()
    else:
        raise ValueError("please ensure that the criterion (loss) is one of the allowed choices")
    
    data_set = TensorDataset(imgs, masks)
    train_set, val_set = random_split(data_set, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(seed))

    return DataLoader(train_set, shuffle = True, **loader_args), DataLoader(val_set, shuffle=False, drop_last=True), n_train, n_val

def save_checkpoint(folder_to_save_data: str,
                    freq_model: UNet,
                    swag_model: SWAG,
                    optimizer: torch.optim,
                    epoch_number: int,
                    tracker_train: None,
                    tracker_eval: None
                    ):
    weights_path = os.path.normpath(os.path.join(folder_to_save_data + 
                                                 "/weights_best_%d.pkl" % str(epoch_number)))
    swag_model_path = os.path.normpath(os.path.join(folder_to_save_data + 
                                                    "/swag_best_%d.pkl" % str(epoch_number)))
    optimizer_path = os.path.normpath(os.path.join(folder_to_save_data + 
                                                   "/optimizer_best_%d.pth" % str(epoch_number)))
    # first, remove old best weights, optimizer and swag model
    try:
        os.remove(weights_path)
        os.remove(swag_model_path)
        os.remove(optimizer_path)
    except FileNotFoundError:
        # will occur for the first time of saving the best performers.
        # in other cases, it does not hurt to skip the file deleting step
        pass

    torch.save(freq_model, weights_path)
    torch.save(swag_model, swag_model_path)
    torch.save(optimizer, optimizer_path)
    # if (tracker_train is not None) and (tracker_eval is not None):
    #     tracker_train.save_data_dict(save_plotter_path)
    #     tracker_eval.save_data_dict(save_plotter_path)
    return None