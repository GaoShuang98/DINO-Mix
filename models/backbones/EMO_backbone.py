from models.backbones.EMO.model.emo import EMO_6M, EMO_1M, EMO_2M, EMO_5M
import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat



class EMO_new(nn.Module):

    def __init__(self, layers_to_crop=[4], model_name='EMO-6M', layer_to_freeze=3):
        super().__init__()
        self.model_name = model_name.lower()  # 将大写转化成小写
        self.layers_to_crop = layers_to_crop
        self.layer_to_freeze = layer_to_freeze

        if '1m' in self.model_name:
            self.EMO_model = EMO_1M()
            self.EMO_model.load_state_dict(
                torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\EMO\resources\EMO_1M\net.pth'))
        elif '2m' in self.model_name:
            self.EMO_model = EMO_2M()
            self.EMO_model.load_state_dict(
                torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\EMO\resources\EMO_2M\net.pth'))
        elif '5m' in self.model_name:
            self.EMO_model = EMO_5M()
            self.EMO_model.load_state_dict(
                torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\EMO\resources\EMO_5M\net.pth'))
        elif '6m' in self.model_name:
            self.EMO_model = EMO_6M()
            self.EMO_model.load_state_dict(
                torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\EMO\resources\EMO_6M\net.pth'))
        else:
            raise NotImplementedError(
                f'Backbone architecture(name:{self.model_name}) not recognized!')

        if self.layer_to_freeze > 0:
            self.EMO_model.stage0.requires_grad_(False)
        if self.layer_to_freeze >= 1:
            self.EMO_model.stage1.requires_grad_(False)
        if self.layer_to_freeze >= 2:
            self.EMO_model.stage2.requires_grad_(False)
        if self.layer_to_freeze >= 3:
            self.EMO_model.stage3.requires_grad_(False)
            # if self.layer_to_freeze == 3:
            #     for i in range(5, len(self.EMO_model.stage3)):
            #         self.EMO_model.stage3[i].requires_grad_(True)
        if self.layer_to_freeze >= 4:
            self.EMO_model.stage4[0].requires_grad_(False)
            self.EMO_model.stage4[1].requires_grad_(False)
            self.EMO_model.stage4[2].drop_path = nn.Sequential()

        if 4 in self.layers_to_crop:
            self.EMO_model.stage4 = nn.Sequential()
        if 3 in self.layers_to_crop:
            self.EMO_model.stage3 = nn.Sequential()
        if 2 in self.layers_to_crop:
            self.EMO_model.stage2 = nn.Sequential()

        self.EMO_model.norm = nn.Sequential()
        self.EMO_model.pre_head = nn.Sequential()
        self.EMO_model.head = nn.Sequential()


    def forward(self, x):
        x = self.EMO_model.forward_features(x)

        return x.to(torch.float32)


if __name__ == '__main__':

    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    import time

    def get_timepc():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def get_net_params(net):
        num_params = 0
        for param in net.parameters():
            if param.requires_grad:
                num_params += param.numel()
        return num_params / 1e6

    bs = 2  # batch size
    reso = 224  # pic size
    # reso = 256
    x = torch.randn(bs, 3, reso, reso).cuda()

    model = EMO_new(layer_to_freeze=2, layers_to_crop=[], model_name='emo-6m').cuda()
    for name, param in model.EMO_model.named_parameters():
        if param.requires_grad:
            print(f'***{name}**')
    model.eval()
    print(model)
    y = model(x)
    print(f"Input shape is {x.shape}")
    print(f"Output shape is {y.shape}")
    # print(y)

    # fn1 = copy.deepcopy(fn)
    # for blk in fn1.stage0:
    # 	blk.attn_pre = False
    # for blk in fn1.stage1:
    # 	blk.attn_pre = False
    # for blk in fn1.stage2:
    # 	blk.attn_pre = False
    # for blk in fn1.stage3:
    # 	blk.attn_pre = False
    # for blk in fn1.stage4:
    # 	blk.attn_pre = False
    # y1 = fn1(x)
    # print(y1['out'])

    # flops = FlopCountAnalysis(fn, torch.randn(1, 3, 224, 224).cuda())
    # flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).cuda())
    # print(flop_count_table(flops, max_depth=3))
    # flops = FlopCountAnalysis(model, x).total() / bs / 1e9
    # params = parameter_count(model)[''] / 1e6
    # with torch.no_grad():
    #     pre_cnt, cnt = 5, 10
    #     for _ in range(pre_cnt):
    #         y = model(x)
    #     t_s = get_timepc()
    #     for _ in range(cnt):
    #         y = model(x)
    #     t_e = get_timepc()
    # print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params,
    #                                                                              bs * cnt / (t_e - t_s)))
    # print(flop_count_table(FlopCountAnalysis(fn, x), max_depth=3))
