
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

# NOTE: you need to download the SPED dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.


# DATASET_ROOT = 'J:\zcl\datasets\spedtest\\'          # 自己电脑路径
# GT_ROOT = 'J:\zcl\zcl_MixVPR-main(train)\datasets\\' # BECAREFUL, this is the ground truth that comes with GSV-Cities

DATASET_ROOT = 'D:\zcl\datasets\spedtest\\'          # 自己电脑路径
GT_ROOT = 'D:\zcl\zcl_MixVPR-main(train)\datasets\\' # BECAREFUL, this is the ground truth that comes with GSV-Cities


# DATASET_ROOT = 'E:\zcl\datasets\spedtest\\'          # 自己电脑路径
# GT_ROOT = 'E:\zcl\zcl_MixVPR-main(train)\datasets\\' # BECAREFUL, this is the ground truth that comes with GSV-Cities

# DATASET_ROOT = 'E:\zcl\datasets\spedtest\\'            # 师兄电脑路径
# GT_ROOT = 'E:\zcl\zcl_MixVPR-main(train)\datasets\\'   # BECAREFUL, this is the ground truth that comes with GSV-Cities


# 判断路径是否存在
path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to SPED dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')

class SpedDataset(Dataset):
    def __init__(self, input_transform=None):
        self.dataroot  =DATASET_ROOT

        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT+'sped/sped_dbImages.npy')
        
        # query images names
        self.qImages = np.load(GT_ROOT+'sped/sped_qImages.npy')
        
        # ground truth
        self.ground_truth = np.load(GT_ROOT+'sped/sped_gt.npy', allow_pickle=True)
        
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


















