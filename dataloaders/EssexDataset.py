
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

# NOTE: you need to download the ESSEX dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.

DATASET_ROOT = '/home/USER/work/VPR-Bench/datasets/ESSEX3IN1_dataset/'
GT_ROOT = '/home/USER/work/gsv-cities/datasets/' # BECAREFUL, this is the ground truth that comes with GSV-Cities

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to ESSEX3IN1 dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')

class EssexDataset(Dataset):
    def __init__(self, input_transform = None):
        self.dataroot = DATASET_ROOT
        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT+'ESSEX/ESSEX_dbImages.npy')
        
        # query images names
        self.qImages = np.load(GT_ROOT+'ESSEX/ESSEX_qImages.npy')
        
        # ground truth
        self.ground_truth = np.load(GT_ROOT+'ESSEX/ESSEX_gt.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)









    