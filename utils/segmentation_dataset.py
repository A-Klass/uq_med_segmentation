import torch
from pathlib import Path
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_dir: str, 
                 masks_dir: str,
                 scale: float = 1.0, 
                 mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        # self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long