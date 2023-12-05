import os
import numpy as np
import torch
from torch import nn
from pathlib import Path
import os.path as path
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils_ import GradCAM, show_cam_on_image, center_crop_img
# from vit_model import vit_base_patch16_224
from models import helper
import utils

DATASET_ROOT = 'D:\python_code\MixVPR(hgs)'

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} is correct')


class VPRModel(nn.Module):  # 继承pytorch-lightning.LightningModule模块
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone 主干网络
                 backbone_arch='dinov2_vitl14',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,

                 # ---- Aggregator 聚合方法
                 agg_arch='MixVPR',  # CosPlace, NetVLAD, GeM
                 agg_config={},

                 # ---- Train HypeParameters 训练超参数
                 lr=0.03,
                 optimizer='sgd',
                 weight_decay=1e-3,
                 momentum=0.9,
                 warmup_steps=500,
                 milestones=[5, 10, 15],
                 lr_mult=0.3,

                 # ----- Loss 损失函数
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch  # 主干网络名称
        self.pretrained = pretrained  # 是否预训练
        self.layers_to_freeze = layers_to_freeze  # 冻结网络层名称
        self.layers_to_crop = layers_to_crop  # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        self.layer1 = layer1
        self.use_cls = use_cls
        self.norm_descs = norm_descs

        self.agg_arch = agg_arch  # CosPlace, NetVLAD, GeM
        self.agg_config = agg_config  # 聚合方法参数

        self.lr = lr  # 学习率
        self.optimizer = optimizer  # 优化器
        self.weight_decay = weight_decay  # 权重衰减 防止过拟合避免梯度爆炸
        self.momentum = momentum  # SGD中的momentum
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name  # 损失函数名称
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        # self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator 获得主干网络和聚合器
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1=self.layer1, use_cls=self.use_cls,
                                            norm_descs=self.norm_descs)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


class ReshapeTransform:
    def __init__(self, model):
        # input_size = model.backbone.dino_model.patch_embed.img_size
        # patch_size = model.backbone.dino_model.patch_embed.patch_size
        # self.h = input_size[0] // patch_size[0]
        # self.w = input_size[1] // patch_size[1]
        self.h = 16
        self.w = 16

    def __call__(self, x):
        result = x
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # result = x.reshape(x.size(0),
        #                     x.size(1),
        #                     self.h,
        #                     self.w,
        #                      )

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    # model = vit_base_patch16_224(
    # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    # weights_path = "./vit_base_patch16_224.pth"
    # model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    model = VPRModel(
        # ---- Encoder
        backbone_arch='dinov2_vitb14',
        # backbone_arch='resnet50',
        pretrained=True,
        # layers_to_freeze=2,
        # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

        layer1=7,  # *层的norm1
        # layer2=28,
        # layer3=31,
        # facet1="value", # 整个output_hook（QKV）
        # facet2="value",
        use_cls=False,
        norm_descs=True,

        agg_arch='DinoMixVPR',  # DinoMixVPR  None
        agg_config={
            'in_channels': 768,
            'in_h': 16,
            'in_w': 16,
            'out_channels': 1024,
            'mix_depth': 2,
            'mlp_ratio': 1,
            'out_rows': 4},  # the output dim will be (out_rows * out_channels) 输出维度为 out_rows * out_channels

        # ---- Train hyperparameters 训练超参数
        lr=0.01,  # 0.0002 for adam, 0.05 for sgd (needs to change according to batch size) 需要根据batch—size调整学习率
        optimizer='sgd',  # sgd, adamw
        # optimizer='adamw',  # sgd, adamw
        weight_decay=0.001,  # 0.001 for sgd and 0 for adam,
        # weight_decay=0,  # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmup_steps=650,
        milestones=[5, 10, 15, 25, 45],
        lr_mult=0.3,

        # ----- Loss functions 损失函数
        # example: ContrastiveLoss 对比损失, TripletMarginLoss 三元组损失, MultiSimilarityLoss 多重相似性损失,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner',  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    weight_path = r'D:\python_code\MixVPR(hgs)\LOGS\dinov2_vitb14\lightning_logs\version_22\checkpoints\dinov2_vitb14_epoch(11)_step(2352)_R1[0.9428]_R5[0.9884]_R10[0.9941]_R1[0.9203]_R5[0.9589]_R10[0.9712].ckpt'
    # model_name = 'dinov2_vitb14'
    # model = torch.hub.load(model_path, model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
    # model.eval()

    checkpoint = torch.load(weight_path, map_location='cuda')  # map_location='cpu'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # state_dict = OrderedDict(
    # 	{k.replace('model.', ''): v for (k, v) in state_dict.items()})
    model_dict_weight = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if
                  k in model_dict_weight}
    model_dict_weight.update(state_dict)
    # 查找缺失和意外得到的预训练模型的权重参数
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")
    # 最终加载模型预训练参数内容
    model.load_state_dict(model_dict_weight)
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    print("\n\n")

    # Since the final classification is done on the class token computed in the last attention block,
    # 由于最终分类是在最后关注块中计算的cls token上进行的
    # the output will not be affected by the 14x14 channels in the last layer.
    # 输出将不受最后一层中的14x14通道的影响。
    # The gradient of the output with respect to them, will be 0!
    # 输出相对于它们的梯度将为0
    # We should chose any layer before the final attention block.
    # 我们应该在最后的注意力块之前选择任何一层

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_name = '@0@00352.3@@@@@148@@@@@@@@.jpg'
    img_dir = r"D:\python_code\MixVPR(hgs)\grad_cam(热力图可视化)\test_imgs"
    img_path = path.join(img_dir, img_name)
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img_show = img
    img_show = img.resize((224, 224))
    img = np.array(img_show, dtype=np.uint8)
    # img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    target_category = 500  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # print(len(model.backbone.dino_model.blocks))
    # for i in range(1, len(model.backbone.dino_model.blocks[10])):
    target_layers = [model.backbone.dino_model.blocks[10]]
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=True,
                  reshape_transform=ReshapeTransform(model)
                  )

    # for j in range(2, 3):
    #     grayscale_cam = cam(input_tensor=input_tensor, target_category=j)
    #     grayscale_cam = grayscale_cam[0, :]
    #     heatmap += grayscale_cam/

    dirname = img_name + 'dinoMix_b14_agg_mix1'
    save_dir = r"D:\python_code\MixVPR(hgs)\grad_cam(热力图可视化)\saved_figs"
    dir = path.join(save_dir, dirname)
    if not os.path.exists(dir):
        os.mkdir(dir)

    for class_num in range(0, 1536):
        # heatmap = np.zeros_like(img[:, :, 0].astype(dtype=np.float32))  # 创建一个新的
        grayscale_cam = cam(input_tensor=input_tensor, target_category=class_num)
        grayscale_cam = grayscale_cam[0, :]
        # heatmap += grayscale_cam
        # heatmap = heatmap / num_classes
        visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)

        plt.imshow(visualization)
        plt.title(f'dinoMix_b14|blocks[10]|target_category={str(class_num)}')
        plt.axis('off')
        plt.savefig(path.join(dir, f'{img_name}{str(class_num)}.jpg'))
        plt.show()
        plt.close()

        # plt.imsave(path.join(dir, f'{img_name}dinoMix_b14_agg_mix1[{str(i)}]_target_category_{str(class_num)}.jpg'),
        #     arr=visualization)


if __name__ == '__main__':
    main()
