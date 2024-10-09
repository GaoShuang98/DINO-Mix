
import re
import torch
import shutil
import logging
import torchscan
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA

import datasets_ws


def get_flops(model, input_shape=(480, 640)):
    """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    assert len(input_shape) == 2, f"input_shape should have len==2, but it's {input_shape}"
    module_info = torchscan.crawl_module(model, (3, input_shape[0], input_shape[1]))
    output = torchscan.utils.format_info(module_info)
    return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model


def resume_model_feature_extract(arg, model):

    # # 加载预训练模型
    weight_path = arg.resume
    checkpoint = torch.load(weight_path, map_location='cuda')   # map_location='cpu'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = OrderedDict({k.replace('aggregator.', 'aggregation.'): v for (k, v) in state_dict.items()})
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

def resume_mixvprmodeleval(args, model):

    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict_weight = model.state_dict()
    # 测试部分（即eval）需要先进行修改掉
    # 记住修改
    # model_dict_weight = model.state_dict()

    # 修改部分
    # 先去掉model.
    state_dict = OrderedDict(
        {k.replace('model.', ''): v for (k, v) in state_dict.items()})
    # 修改layer1.为4.
    state_dict = OrderedDict(
        {k.replace('layer1.', '4.'): v for (k, v) in state_dict.items()})
    # 修改layer2.为5.
    state_dict = OrderedDict(
        {k.replace('layer2.', '5.'): v for (k, v) in state_dict.items()})
    # 修改layer3.为6.
    state_dict = OrderedDict(
        {k.replace('layer3.', '6.'): v for (k, v) in state_dict.items()})
    # 修改前6列的key值
    state_dict = OrderedDict(
        [('backbone.0.weight', v) if k == 'backbone.conv1.weight' else (k, v) for k, v in state_dict.items()])
    state_dict = OrderedDict(
        [('backbone.1.weight', v) if k == 'backbone.bn1.weight' else (k, v) for k, v in state_dict.items()])
    state_dict = OrderedDict(
        [('backbone.1.bias', v) if k == 'backbone.bn1.bias' else (k, v) for k, v in state_dict.items()])
    state_dict = OrderedDict(
        [('backbone.1.running_mean', v) if k == 'backbone.bn1.running_mean' else (k, v) for k, v in
         state_dict.items()])
    state_dict = OrderedDict(
        [('backbone.1.running_var', v) if k == 'backbone.bn1.running_var' else (k, v) for k, v in
         state_dict.items()])
    state_dict = OrderedDict(
        [('backbone.1.num_batches_tracked', v) if k == 'backbone.bn1.num_batches_tracked' else (k, v) for k, v in
         state_dict.items()])

    # 修改backbone.为module.backbone.
    state_dict = OrderedDict(
        {k.replace('backbone.', 'module.backbone.'): v for (k, v) in state_dict.items()})

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

    # 修改aggregator.为module.aggregation.
    state_dict = OrderedDict(
        {k.replace('aggregator.', 'module.aggregation.'): v for (k, v) in state_dict.items()})

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

def resume_model_feature_extract_mix(arg, model):

    # # 加载预训练模型
    weight_path = arg.resume
    checkpoint = torch.load(weight_path, map_location='cuda')   # map_location='cpu'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = OrderedDict({k.replace('aggregator.', 'aggregation.'): v for (k, v) in state_dict.items()})
    state_dict = OrderedDict({k.replace('model.', 'Sequential.'): v for (k, v) in state_dict.items()})
    state_dict = OrderedDict({k.replace('model.', ''): v for (k, v) in state_dict.items()})
    state_dict = OrderedDict({k.replace('layer', ''): v for (k, v) in state_dict.items()})

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


def resume_mixvprtrainmodel(args, model):
    # # 加载hgs的dino-mix预训练模型
    weight_path = args.resume
    checkpoint = torch.load(weight_path, map_location=args.device)  # map_location='cpu'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = OrderedDict(
    	{k.replace('aggregator.', 'aggregation.'): v for (k, v) in state_dict.items()})

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

def resume_dino_model(args, model):
    # # 加载预训练模型
    weight_path = args.resume
    checkpoint = torch.load(weight_path, map_location=args.device)  # map_location='cpu'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = OrderedDict(
    	{k.replace('aggregator.', 'aggregation.'): v for (k, v) in state_dict.items()})

    model_dict_weight = model.backbone.dino_model.state_dict()

    state_dict = {k: v for k, v in state_dict.items() if
                  k in model_dict_weight}
    model_dict_weight.update(state_dict)

    # 查找缺失和意外得到的预训练模型的权重参数
    missing_keys, unexpected_keys = model.backbone.dino_model.load_state_dict(state_dict, strict=False)

    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")
    # 最终加载模型预训练参数内容
    model.backbone.dino_model.load_state_dict(model_dict_weight)
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    print("\n\n")

    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters
    加载模型、优化器和其它训练参数
    """
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, " \
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num


def compute_pca(args, model, pca_dataset_folder, full_features_dim):
    """
    计算PCA
    """
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(args, args.datasets_folder, pca_dataset_folder)
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i*args.infer_batch_size >= len(pca_features): break
            features = model(images).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca

