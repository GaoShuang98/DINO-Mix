import lightning_lite
import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import utils

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper
from collections import OrderedDict
from DINO_Mix import VPRModel


if __name__ == '__main__':
    lightning_lite.utilities.seed.seed_everything(seed=190223, workers=True)

    datamodule = GSVCitiesDataModule(
        batch_size=80,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,  # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=7,
        show_data_stats=True,
        val_set_names=['pitts30k_val', 'pitts30k_test'],  # pitts30k_val, pitts30k_test, msls_val
    )

    # examples of backbones
    # resnet18, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
    # efficientnet_b0, efficientnet_b1, efficientnet_b2
    # swinv2_base_window12to16_192to256_22kft1k
    # dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14

    model = VPRModel(
        # ---- Encoder
        backbone_arch='dinov2_vitb14',
        # backbone_arch='resnet50',
        pretrained=True,
        # layers_to_freeze=2,
        # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc

        layer1=8,  # *层的norm1
        # layer2=28,
        # layer3=31,
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

        agg_arch='DinoMixVPR',  #   DinoMixVPR  None
        agg_config={
                    'in_channels': 768,
                    'in_h': 16,
                    'in_w': 16,
                    'out_channels': 1024,
                    'mix_depth': 2,
                    'mlp_ratio': 1,
                    'out_rows': 4},  # the output dim will be (out_rows * out_channels) 输出维度为 out_rows * out_channels

        # ---- Train hyperparameters
        lr=0.005,  # 0.0002 for adam, 0.05 for sgd (needs to change according to batch size)
        optimizer='sgd',  # sgd, adamw
        weight_decay=0.001,  # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmup_steps=650,
        milestones=[5, 10, 15, 25, 45],
        lr_mult=0.3,

        # # ###########################################################################################################
        # # Fine-tune training parameters (start with the best)
        # # ---- Train hyperparameters     # 0.05   0.01   0.03
        # # lr=0.05,  0.05                 # 当batch_size改为100时，对于sgd优化器需要修改成为0.03
        # lr=0.00002,
        # # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)   0.05改为0.0125   0.05   0.03   0.01   0.005
        # optimizer='adam',  # sgd, adamw, adam
        # weight_decay=0,    # 0.001 for sgd and 0 for adam,    0.001
        # momentum=0.9,
        # # warmpup_steps=0,   # 500 100 125  0   500改为0，主要原因是在训练好的模型上面进行加载训练，不在需要进行warmup，直接进行训练就行
        # # warmpup_steps=0, # 500 100 125  0   500改为0，主要原因是在训练好的模型上面进行加载训练，不在需要进行warmup，直接进行训练就行
        # milestones=[5, 10, 15, 25, 45],  # 5*1 5*2 5*3 5*5 5*9
        # # milestones=[5, 15, 25, 45],  # 5*1 5*2 5*3 5*5 5*9
        # # milestones=[8, 16, 24, 32, 48],     # 8*1 8*2 8*3 8*4 8*6
        # # milestones=[10, 20, 30, 40, 45],  # 8*1 8*2 8*3 8*4 8*6
        # lr_mult=0.3,
        #
        # # # 0.05*0.3=0.015    0.05*0.3*0.3=0.0045     0.05*0.3*0.3*0.3=0.00135    0.05*0.3*0.3*0.3*0.3=0.000405
        # # # 0.05 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3 = 0.0001215
        #
        # #############################################################################################################

        # ----- Loss functions 损失函数
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner',  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    print(model)


    # ###########################################resume the trained model #############################################
    # # retrain where annotations are needed start here
    # # resume the model
    # weight_path = r"\DINO-Mix\dinov2_vitb14.ckpt"
    # checkpoint = torch.load(weight_path, map_location='cuda')   # map_location='cpu'
    # if 'state_dict' in checkpoint:
    #     state_dict = checkpoint['state_dict']
    # else:
    #     state_dict = checkpoint
    # # state_dict = OrderedDict(
    # # 	{k.replace('model.', ''): v for (k, v) in state_dict.items()})
    # model_dict_weight = model.state_dict()
    # state_dict = {k: v for k, v in state_dict.items() if
    #               k in model_dict_weight}
    # model_dict_weight.update(state_dict)
    # # Find missing and unexpected weight parameters for pretrained models
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")
    # # Finally, load the content of the model pre-trained parameters
    # model.load_state_dict(model_dict_weight)
    # print("\n\033[1;33;44m Reminder: It is normal for the head part to not be loaded, and it is wrong for the backbone part to not be loaded。\033[0m")
    # print("\n\n")
    # # ###########################################################################################


    # model params saving using Pytorch Lightning
    # we save the best 3 models according to Recall@1 on pittsburgh val
    checkpoint_cb = ModelCheckpoint(
        monitor='pitts30k_test/R1',
        filename=f'{model.encoder_arch}' +
                 '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]_R10[{pitts30k_val/R10:.4f}]_R1[{pitts30k_test/R1:.4f}]_R5[{pitts30k_test/R5:.4f}]_R10[{pitts30k_test/R10:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=15,
        mode='max', )

    # ------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[1],
        # strategy='DP',  # use the stand alone multi sim mode
        # distributed_backend='dp',
        default_root_dir=f'./LOGS/{model.encoder_arch}',
        # w can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=60,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=25,
        # fast_dev_run=True,  # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
        # limit_train_batches=10,   # only 10 batches are used for training
        # limit_val_batches=0.5,  # only 5 batches are used for validation
        # limit_train_batches=2,   # Each training epoch only runs one-tenth of the data
        benchmark=True,  # If your model's input size remains the same, you can set cudnn.benchmark to True, which will speed up the training, and if the input size is not fixed, it will slow down the training.
        accumulate_grad_batches=4,  # Gradient accumulation, four batches of gradients at a time (batch size is equivalent to a four-fold increase) // In certain cases, the data is too large to fit into a batch, or a large batch size is needed and there are not so many GPUs to use, then a large batch size can be simulated by accumulating gradients
        gradient_clip_val=0.5,  # Clipping the gradient value > 0.5 prevents the gradient from exploding
        gradient_clip_algorithm='value'
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
