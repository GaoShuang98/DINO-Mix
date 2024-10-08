dependencies = [
    'torch',
    'torchvision',
    'pytorch_lightning',
 ]

import torch
from models import helper

DEFAULT_AGG_CONFIG = {
    'in_channels': 768,
    'in_h': 16,
    'in_w': 16,
    'out_channels': 1024,
    'mix_depth': 2,
    'mlp_ratio': 1,
    'out_rows': 4
}


class VPRModel(torch.nn.Module):
    """
    VPR Model with a backbone and an aggregator.

    Args:
        backbone_arch (str): Architecture of the backbone.
        pretrained (bool): Whether to use a pretrained backbone.
        layer1 (int): Layer index for backbone.
        use_cls (bool): Whether to use classification token.
        norm_descs (bool): Whether to normalize descriptors.
        agg_arch (str): Architecture of the aggregator (e.g., DinoMixVPR).
        agg_config (dict): Configuration for the aggregator.
    """
    def __init__(self,
                 # ---- Backbone
                 backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layer1=7,
                 use_cls=False,
                 norm_descs=True,
                 # ---- Aggregator
                 agg_arch='DinoMixVPR',
                 agg_config={},

                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layer1 = layer1
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.backbone = helper.get_backbone(self.backbone_arch, self.pretrained, layer1=self.layer1,  use_cls=self.use_cls, norm_descs=self.norm_descs)
        self.aggregator = helper.get_aggregator(self.agg_arch, self.agg_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

def dino_mix(pretrained=True, **kwargs):
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        pretrained=pretrained,
        layer1=7,
        use_cls=False,
        norm_descs=True,
        agg_arch='DinoMixVPR',
        agg_config=DEFAULT_AGG_CONFIG,
        **kwargs
    )
    if pretrained:
        checkpoint_url = "https://github.com/GaoShuang98/DINO-Mix/releases/download/v1.0.0/dinov2_vitb14_mix.ckpt"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
        model.load_state_dict(state_dict)
    return model