import torch
from torch.utils.data import Dataset
import nibabel as nib


class Dataset_Pro(Dataset):
    def __init__(self,file_path):
        super(Dataset_Pro, self).__init__()

