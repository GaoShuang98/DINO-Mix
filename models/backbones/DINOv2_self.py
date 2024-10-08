# --<utf-8>--


import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2_self(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer1: int = 39, use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layer1 = layer1

        self.pretrained = pretrained
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            if self.layer1 > 39:
                print('Please confirm the correctness of the layer! The highest block layer of vitg14 is 39 layers')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            if self.layer1 > 23:
                print('Please confirm the correctness of the layer! The highest block layer of vitl14 is 23 layers')
                exit()
        elif 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            if self.layer1 > 11:
                print('Please confirm the correctness of the layer! The highest block layer of VITB14 is 11 layers')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            if self.layer1 > 11:
                print('Please confirm the correctness of the layer! The highest block layer of vits14 is 11 layers')
                exit()
        else:
            print(f'The model name definition is incorrect, please check model_name:{self.dino_model}')


        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)

        self.dino_model.norm = nn.Sequential()
        self.dino_model.head = nn.Sequential()


    def forward(self, x, masks=None):

        x = self.dino_model.forward_features(x)
        x = x['x_norm_patchtokens']  # 取无cls的输出
        bs, f, c = x.shape
        x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # 拆分通道，转换成特征图形式
        return x.permute(0, 3, 1, 2)

# -----------------------------------------------debug-----------------------------------------------------------------/

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 3, 224, 224).to('cuda')
    model = DinoV2_self(model_name='dinov2_vitb14', layer1=11, facet1="value", use_cls=False, norm_descs=True, device="cuda", pretrained=True)
    # torch.onnx.export(model.dino_model, torch.randn(1, 3, 224, 224), 'dinov2_vitl14.onnx', do_constant_folding=True, verbose=False)

    print(model)
    # print(model.dino_model.cls_token)
    # print(model.dino_model.pos_embed)
    # print(model.dino_model.mask_token)
    for name, param in model.dino_model.named_parameters():
        if param.requires_grad:
            print(f'***{name}**')

    print('-' * 70)
    summary(model, (3, 224, 224), 1, 'cuda')
    print('-' * 70)

    r = model(x)

    print_nb_params(model)

    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')


if __name__ == '__main__':
    main()


