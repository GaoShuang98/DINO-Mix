
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.

# DATASET_ROOT = '../datasets/msls_val/'     # E:\usefuldata\msls

# 修改五
# DATASET_ROOT = 'E:\\usefuldata\msls_val\\'        # 自己电脑

# DATASET_ROOT = 'E:\\usefuldata\msls\\'      # 师兄电脑

# DATASET_ROOT = 'D:\python_code\MixVPR\datasets\msls_val\\'      # 师兄电脑

# DATASET_ROOT = 'E:\zcl\datasets\msls_val\\'      # 师兄电脑 D:\python_code\MixVPR\datasets\msls_val

# DATASET_ROOT = 'D:\python_code\MixVPR\datasets\msls_val\\'      # 师兄电脑 D:\python_code\MixVPR\datasets\msls_val


# DATASET_ROOT = 'J:\zcl\datasets\msls_val\\'      # 自己电脑     只在自己电脑上面进行测试    E:\usefuldata\msls_val
# # DATASET_ROOT = r'E:\usefuldata\msls_val\\'
#
# # DATASET_ROOT = 'E:\zcl\datasets\msls_val\\'      # 师兄电脑
# GT_ROOT = 'J:\zcl\zcl_MixVPR-main(train)\datasets\\'
# # GT_ROOT = r'I:\zclzhijieshiyong\test\zcl_MixVPR-main(train)\datasets\\'

DATASET_ROOT = 'D:\zcl\datasets\msls_val\\'      # 自己电脑     只在自己电脑上面进行测试    E:\usefuldata\msls_val
# DATASET_ROOT = r'E:\usefuldata\msls_val\\'

# DATASET_ROOT = 'E:\zcl\datasets\msls_val\\'      # 师兄电脑
GT_ROOT = 'D:\zcl\zcl_MixVPR-main(train)\datasets\\'
# GT_ROOT = r'I:\zclzhijieshiyong\test\zcl_MixVPR-main(train)\datasets\\'




# 判断路径是否存在
path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to mapillary_sls dataset is correct')

if not path_obj.joinpath('train_val'):
    raise Exception(f'Please make sure the directory train_val from mapillary_sls dataset is situated in the directory {DATASET_ROOT}')

class MSLS(Dataset):
    def __init__(self, input_transform=None):

        self.dataroot = DATASET_ROOT
        
        self.input_transform = input_transform
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        # self.dbImages = np.load('J:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_dbImages.npy')   # 数据库图像
        # self.dbImages = np.load('E:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_dbImages.npy')  # 数据库图像
        self.dbImages = np.load(GT_ROOT+'msls_val\msls_val_dbImages.npy')

        # hard coded query image names.
        # self.qImages = np.load('J:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_qImages.npy')     # 查询库图像
        # self.qImages = np.load('E:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_qImages.npy')     # 查询库图像
        self.qImages = np.load(GT_ROOT+'msls_val\msls_val_qImages.npy')

        # hard coded index of query images
        # self.qIdx = np.load('J:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_qIdx.npy')           # 查询库索引
        # self.qIdx = np.load('E:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_qIdx.npy')           # 查询库索引
        self.qIdx = np.load(GT_ROOT+'msls_val\msls_val_qIdx.npy')

        # hard coded groundtruth (correspondence between each query and its matches)
        # self.ground_truth = np.load('J:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_pIdx.npy', allow_pickle=True)    # 验证索引
        # self.ground_truth = np.load('E:\zcl\zcl_MixVPR-main(train)\datasets\msls_val\msls_val_pIdx.npy', allow_pickle=True)    # 验证索引
        self.ground_truth = np.load(GT_ROOT+'msls_val\msls_val_pIdx.npy',
                                    allow_pickle=True)

        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index



    def __len__(self):

        return len(self.images)













