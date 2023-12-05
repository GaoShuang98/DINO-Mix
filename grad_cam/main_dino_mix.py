import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils_ import GradCAM, show_cam_on_image, center_crop_img
from DINO_Mix import VPRModel
from collections import OrderedDict

class ReshapeTransform:
    def __init__(self, model):
        # input_size = model.backbone.dino_model.patch_embed.img_size
        # patch_size = model.backbone.dino_model.patch_embed.patch_size
        # self.h = input_size[0] // patch_size[0]
        # self.w = input_size[1] // patch_size[1]
        self.h = 16
        self.w = 16

    def __call__(self, x):
        # result = x
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        result = result.permute(0, 3, 1, 2)
        return result

def resum_vprmodel(resume_dir, model):

    # # 加载预训练模型
    checkpoint = torch.load(resume_dir, map_location='cuda')  # map_location='cpu'
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
    return model


def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]
    resume_dir = 'dinov2_vitb14.ckpt'
    model_net = VPRModel(
        # ---- Encoder
        backbone_arch='dinov2_vitb14',
        pretrained=True,
        layer1=8,  # *层的norm1
        use_cls=False,
        norm_descs=True,

        # ---- Aggregator
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 768,
        #             'out_dim': 4096},
        # agg_arch='GeM',
        # agg_config={'p': 3},

        # agg_arch='ConvAP',
        # agg_config={'in_channels': 2048,
        #             'out_channels': 2048},

        agg_arch='DinoMixVPR',
        agg_config={
                    'in_channels': 768,
                    'in_h': 16,
                    'in_w': 16,
                    'out_channels': 1024,
                    'mix_depth': 2,
                    'mlp_ratio': 1,
                    'out_rows': 4},  # the output dim will be (out_rows * out_channels) 输出维度为 out_rows * out_channels

        # ---- Train hyperparameters 训练超参数
        lr=0.005,  # 0.0002 for adam, 0.05 for sgd (needs to change according to batch size) 需要根据batch—size调整学习率
        optimizer='sgd',  # sgd, adamw
        weight_decay=0.001,  # 0.001 for sgd and 0 for adam,
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

    model = resum_vprmodel(resume_dir, model_net)
    print(model)

    target_layers = [model.backbone.dino_model.blocks[10]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                         transforms.Resize([224, 224])]
                                        )
    batch_dir = r'C:\Users\Administrator\Desktop\hgs图像检索文件\论文中待检索图像'
    grad_cam_batch(batch_dir=batch_dir, model=model, target_layers=target_layers)



def grad_cam_single(img_path, model, target_layers):

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                         transforms.Resize([224, 224])])

    img_name = os.path.split(img_path)[-1]

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img_show = img
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)


    target_category = None  # best
    # target_category = 254  # self choice

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=True,
                  reshape_transform=ReshapeTransform(model)
                  )

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    img = img_show.resize((224, 224))

    img = np.array(img, dtype=np.uint8)
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    # plt.savefig(os.path.join(r'D:\python_code\MixVPR(hgs)\grad_cam(热力图可视化)\热力图结果-不同方法\MixVPR',
    #                          'MixVPR' + img_name))
    plt.close()

def grad_cam_batch(batch_dir, model, target_layers):

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                         transforms.Resize([224, 224])])

    img_list = os.listdir(batch_dir)
    for img_name in img_list:

        # load image
        img_path = os.path.join(batch_dir, img_name)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img_show = img
        img = np.array(img, dtype=np.uint8)
        # img = center_crop_img(img, 224)

        # [C, H, W]
        img_tensor = data_transform(img)
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = 690


        cam = GradCAM(model=model,
                      target_layers=target_layers,
                      use_cuda=True,
                      reshape_transform=ReshapeTransform(model)
                      )

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        img = img_show.resize((224, 224))
        img = np.array(img, dtype=np.uint8)
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(os.path.join(r'D:\python_code\MixVPR(hgs)\grad_cam(热力图可视化)\热力图结果-不同方法\dino-mix', 'dino-mix'+img_name))
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()
