import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils_ import GradCAM, show_cam_on_image, center_crop_img
from main import VPRModel
from collections import OrderedDict


def resum_vprmodel(resume_dir, model):
    checkpoint = torch.load(resume_dir, map_location='cuda')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict_weight = model.state_dict()
    # 测试部分（即eval）需要先进行修改掉
    # 记住修改
    # 修改部分
    # 先去掉model.
    # state_dict = OrderedDict(
    #     {k.replace('model.', ''): v for (k, v) in state_dict.items()})
    # # 修改layer1.为4.
    # state_dict = OrderedDict(
    #     {k.replace('layer1.', '4.'): v for (k, v) in state_dict.items()})
    # # 修改layer2.为5.
    # state_dict = OrderedDict(
    #     {k.replace('layer2.', '5.'): v for (k, v) in state_dict.items()})
    # # 修改layer3.为6.
    # state_dict = OrderedDict(
    #     {k.replace('layer3.', '6.'): v for (k, v) in state_dict.items()})
    # # 修改前6列的key值
    # state_dict = OrderedDict(
    #     [('backbone.0.weight', v) if k == 'backbone.conv1.weight' else (k, v) for k, v in state_dict.items()])
    # state_dict = OrderedDict(
    #     [('backbone.1.weight', v) if k == 'backbone.bn1.weight' else (k, v) for k, v in state_dict.items()])
    # state_dict = OrderedDict(
    #     [('backbone.1.bias', v) if k == 'backbone.bn1.bias' else (k, v) for k, v in state_dict.items()])
    # state_dict = OrderedDict(
    #     [('backbone.1.running_mean', v) if k == 'backbone.bn1.running_mean' else (k, v) for k, v in
    #      state_dict.items()])
    # state_dict = OrderedDict(
    #     [('backbone.1.running_var', v) if k == 'backbone.bn1.running_var' else (k, v) for k, v in
    #      state_dict.items()])
    # state_dict = OrderedDict(
    #     [('backbone.1.num_batches_tracked', v) if k == 'backbone.bn1.num_batches_tracked' else (k, v) for k, v in
    #      state_dict.items()])
    #
    # # 修改backbone.为module.backbone.
    # state_dict = OrderedDict(
    #     {k.replace('backbone.', 'module.backbone.'): v for (k, v) in state_dict.items()})

    # # 修改attention.为module.attention.
    # state_dict = OrderedDict(
    #     {k.replace('attention.', 'module.attention.'): v for (k, v) in state_dict.items()})

    # # 修改attention.为module.attention.
    # state_dict = OrderedDict(
    #     {k.replace('rga_att2.', 'module.6.'): v for (k, v) in state_dict.items()})
    # state_dict = OrderedDict(
    #     {k.replace('maxpoolrga_att2.', 'module.7.'): v for (k, v) in state_dict.items()})
    # # 修改attention.为module.attention.
    # state_dict = OrderedDict(
    #     {k.replace('rga_att3.', 'module.9.'): v for (k, v) in state_dict.items()})

    # # 修改featurefusion.为module.featurefusion.
    # state_dict = OrderedDict(
    #     {k.replace('featurefusion.', 'module.featurefusion.'): v for (k, v) in state_dict.items()})

    # # 修改aggregator.为module.aggregation.
    # state_dict = OrderedDict(
    #     {k.replace('aggregator.', 'module.aggregation.'): v for (k, v) in state_dict.items()})

    # state_dict = {k: v for k, v in state_dict.items() if
    #               k in model_dict_weight}
    #
    # # 将预训练好的模型的值，更新到自己模型的dict中
    # model_dict_weight.update(state_dict)
    #
    # # 查找缺失和意外得到的预训练权重参数
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")
    #
    # # # 最终模型的参数
    # # model.load_state_dict(model_dict_weight)
    #
    # # 训练模式下需要先将其注释掉
    # # if list(state_dict.keys())[0].startswith('module'):
    # #     state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    # #

    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})

    state_dict = {k: v for k, v in state_dict.items() if
                  k in model_dict_weight}
    # 将预训练好的模型的值，更新到自己模型的dict中
    model_dict_weight.update(state_dict)

    # 查找缺失和意外得到的预训练权重参数
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")

    model.load_state_dict(model_dict_weight)

    # model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    return model

def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]
    resume_dir = r'D:\python_code\MixVPR(hgs)\LOGS\resnet50\resnet50_convap.ckpt'
    model_net = VPRModel(
        # ---- Encoder
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

        # ---- Aggregator
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 2048,
        #             'out_dim': 2048},
        # agg_arch='GeM',
        # agg_config={'p': 3},

        agg_arch='ConvAP',
        agg_config={'in_channels': 1024,
                    'out_channels': 1024},

        # agg_arch='MixVPR',
        # agg_config={'in_channels': 1024,
        #             'in_h': 20,
        #             'in_w': 20,
        #             'out_channels': 1024,
        #             'mix_depth': 4,
        #             'mlp_ratio': 1,
        #             'out_rows': 4},  # the output dim will be (out_rows * out_channels) 输出维度为 out_rows * out_channels

        # ---- Train hyperparameters 训练超参数
        lr=0.05,  # 0.0002 for adam, 0.05 for sgd (needs to change according to batch size) 需要根据batch—size调整学习率
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

    target_layers = [model.backbone.model.layer3]

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
                                         transforms.Resize([320, 320])]
                                        )
    batch_dir = r'C:\Users\Administrator\Desktop\hgs图像检索文件\论文中待检索图像'
    grad_cam_batch(batch_dir=batch_dir, model=model, target_layers=target_layers)



def grad_cam_single(img_path, model, target_layers):
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                         transforms.Resize([320, 320])]
                                        )
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

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = None  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    img = img_show.resize((320, 320))
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
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                         transforms.Resize([320, 320])]
                                        )
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

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = None  # tabby, tabby cat
        # target_category = 254  # pug, pug-dog

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        img = img_show.resize((320, 320))
        img = np.array(img, dtype=np.uint8)
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.savefig(os.path.join(r'D:\python_code\MixVPR(hgs)\grad_cam(热力图可视化)\热力图结果-不同方法\convap', 'convap'+img_name))
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()
