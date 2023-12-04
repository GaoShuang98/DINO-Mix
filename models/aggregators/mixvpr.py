import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchsummary import summary


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()
        # self.in_h = in_h
        # self.in_w = in_w
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        # 定义一个Sequential容器，用于叠加FeatureMixerLayer
        self.mix = nn.Sequential(*[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)  # hw输入尺寸，out_rows输出尺寸

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)  # Feature-Mixer模块
        x = x.permute(0, 2, 1)  # 将数据转换成 0块 2行1列
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)  # 将数据转换成 0块 2行1列
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)  # 将x展平,并正则化
        return x


# ----------------------------------debug---------------------------------------------

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params / 1e6:.3}M')


def main():
    x = torch.randn(1, 1024, 20, 20).to(torch.device('cuda'))
    agg = MixVPR(
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=1024,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=4).to(torch.device('cuda'))

    # print_nb_params(agg)
    output = agg(x)
    # summary(agg, input_size=(320, 7, 7), batch_size=1, device='cuda')
    print(output.shape)


if __name__ == '__main__':
    main()

