from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from main_block.capsule_layers import ConvSlimCapsule3D, MarginLoss
from main_block.efficientViT3D import EfficientViTBlock3D, PatchMerging3D, PatchExpand3D
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import ConfusionMatrixMetric, DiceMetric, SurfaceDistanceMetric
from monai.networks import one_hot
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from torch import nn

import warnings

warnings.filterwarnings("ignore")

torch.set_printoptions(profile="full")


# Pytorch Lightning module
class EffiViTCaps3D(pl.LightningModule):
    def __init__(
            self,
            in_channels=2,
            out_channels=4,
            lr_rate=2e-4,
            rec_loss_weight=0.1,
            margin_loss_weight=1.0,
            class_weight=None,
            share_weight=False,
            sw_batch_size=128,
            cls_loss="CE",
            val_patch_size=(32, 32, 32),
            overlap=0.75,
            connection="skip",
            val_frequency=100,
            weight_decay=2e-6,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = self.hparams.in_channels
        self.out_channels = self.hparams.out_channels
        self.share_weight = self.hparams.share_weight
        self.connection = self.hparams.connection

        self.lr_rate = self.hparams.lr_rate
        self.weight_decay = self.hparams.weight_decay

        self.cls_loss = self.hparams.cls_loss
        self.margin_loss_weight = self.hparams.margin_loss_weight
        self.rec_loss_weight = self.hparams.rec_loss_weight
        self.class_weight = self.hparams.class_weight

        # Defining losses
        self.classification_loss1 = MarginLoss(class_weight=self.class_weight, margin=0.2)

        if self.cls_loss == "DiceCE":
            self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, ce_weight=self.class_weight)
        elif self.cls_loss == "CE":
            self.classification_loss2 = DiceCELoss(
                softmax=True, to_onehot_y=True, ce_weight=self.class_weight, lambda_dice=0.0
            )
        elif self.cls_loss == "Dice":
            self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, lambda_ce=0.0)
        self.reconstruction_loss = nn.MSELoss(reduction="none")

        self.val_frequency = self.hparams.val_frequency
        self.val_patch_size = self.hparams.val_patch_size
        self.sw_batch_size = self.hparams.sw_batch_size
        self.overlap = self.hparams.overlap

        # Building model
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Convolution(
                            dimensions=3,
                            in_channels=self.in_channels,
                            out_channels=16,
                            kernel_size=5,
                            strides=1,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (
                        "conv2",
                        Convolution(
                            dimensions=3,
                            in_channels=16,
                            out_channels=32,
                            kernel_size=5,
                            strides=1,
                            dilation=2,
                            padding=4,
                            bias=False,
                        ),
                    ),
                    (
                        "conv3",
                        Convolution(
                            dimensions=3,
                            in_channels=32,
                            out_channels=64,
                            kernel_size=5,
                            strides=1,
                            padding=4,
                            dilation=2,
                            bias=False,
                            act="tanh",
                        ),
                    ),
                ]
            )
        )

        self._build_encoder()
        self._build_decoder()
        self._build_reconstruct_branch()
        self._build_3D_EfficientViTBlock()

        # For validation
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=self.out_channels)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=self.out_channels)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
        self.precision_metric = ConfusionMatrixMetric(
            include_background=False, metric_name="precision", compute_sample=True, reduction="mean_batch", get_not_nans=False
        )
        self.recall_metric = ConfusionMatrixMetric(
            include_background=False, metric_name="recall", compute_sample=True, reduction="mean_batch", get_not_nans=False
        )

        self.example_input_array = torch.rand(1, self.in_channels, 32, 32, 32)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EffiViTCaps3D")
        # Architecture params
        parser.add_argument("--in_channels", type=int, default=2)
        parser.add_argument("--out_channels", type=int, default=4)
        parser.add_argument("--share_weight", type=int, default=1)
        parser.add_argument("--connection", type=str, default="skip")

        # Validation params
        parser.add_argument("--val_patch_size", nargs="+", type=int, default=[32, 32, 32])
        parser.add_argument("--val_frequency", type=int, default=1000)
        parser.add_argument("--sw_batch_size", type=int, default=16)
        parser.add_argument("--overlap", type=float, default=0.75)

        # Loss params
        parser.add_argument("--margin_loss_weight", type=float, default=0.1)
        parser.add_argument("--rec_loss_weight", type=float, default=1e-1)
        parser.add_argument("--cls_loss", type=str, default="CE")

        # Optimizer params
        parser.add_argument("--lr_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=2e-6)
        return parent_parser, parser

    def forward(self, x):
        # Contracting
        x = self.feature_extractor(x)
        # fe_0 = self.feature_extractor.conv1(x)
        # fe_1 = self.feature_extractor.conv2(fe_0)
        # x = self.feature_extractor.conv3(fe_1)

        conv_1_1 = x

        # conv_2_1 = self.encoder_convs[0](conv_1_1)
        conv_2_1 = self.patchMergingblock_1(conv_1_1)
        conv_2_1 = self.relu(conv_2_1)
        conv_2_1 = self.efficientViT3Dblock_encoder_2(conv_2_1)

        conv_3_1 = self.patchMergingblock_2(conv_2_1)
        # conv_3_1 = self.encoder_convs[1](conv_2_1)
        conv_3_1 = self.relu(conv_3_1)
        conv_3_1 = self.efficientViT3Dblock_encoder_3(conv_3_1)

        conv_3_1_reshaped = conv_3_1.view(-1, 8, 16, conv_3_1.shape[-1], conv_3_1.shape[-1], conv_3_1.shape[-1])

        x = self.encoder_conv_caps[0](conv_3_1_reshaped.contiguous())
        # conv_cap_4_1 = x
        conv_cap_4_1 = self.encoder_conv_caps[1](x)

        shape = conv_cap_4_1.size()
        conv_cap_4_1 = conv_cap_4_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        conv_cap_4_1 = self.efficientViT3Dblock_bottleneck(conv_cap_4_1)

        # Expanding
        if self.connection == "skip":
            # ###########################################################################################
            # x = self.patchExpandingblock_3(conv_cap_4_1)
            # x = torch.cat((x, conv_3_1), dim=1)
            # x = self.efficientViT3Dblock_decoder_3(x)
            # x = self.relu(x)
            # x = self.patchExpandingblock_2(x)
            # x = torch.cat((x, conv_2_1), dim=1)
            # x = self.efficientViT3Dblock_decoder_2(x)
            # x = self.relu(x)
            # x = self.patchExpandingblock_1(x)
            # x = torch.cat((x, conv_1_1), dim=1)
            # ###########################################################################################
            x = self.decoder_conv[0](conv_cap_4_1)
            x = torch.cat((x, conv_3_1), dim=1)
            x = self.decoder_conv[1](x)
            x = self.efficientViT3Dblock_decoder_3(x)
            x = self.decoder_conv[2](x)
            x = torch.cat((x, conv_2_1), dim=1)
            x = self.decoder_conv[3](x)
            x = self.efficientViT3Dblock_decoder_2(x)
            x = self.decoder_conv[4](x)
            x = torch.cat((x, conv_1_1), dim=1)

            # extend decover and skip connection
            # x = self.add_deconvs[0](x)
            # x = torch.cat((x, fe_1), dim=1)
            # x = self.add_deconvs[1](x)
            # x = torch.cat((x, fe_0), dim=1)

        logits = self.decoder_conv[5](x)

        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        # Contracting
        x = self.feature_extractor(images)
        # fe_0 = self.feature_extractor.conv1(x)
        # fe_1 = self.feature_extractor.conv2(fe_0)
        # x = self.feature_extractor.conv3(fe_1)

        conv_1_1 = x

        # conv_2_1 = self.encoder_convs[0](conv_1_1)
        conv_2_1 = self.patchMergingblock_1(conv_1_1)
        conv_2_1 = self.relu(conv_2_1)
        conv_2_1 = self.efficientViT3Dblock_encoder_2(conv_2_1)

        conv_3_1 = self.patchMergingblock_2(conv_2_1)
        # conv_3_1 = self.encoder_convs[1](conv_2_1)
        conv_3_1 = self.relu(conv_3_1)
        conv_3_1 = self.efficientViT3Dblock_encoder_3(conv_3_1)

        conv_3_1_reshaped = conv_3_1.view(-1, 8, 16, conv_3_1.shape[-1], conv_3_1.shape[-1], conv_3_1.shape[-1])

        x = self.encoder_conv_caps[0](conv_3_1_reshaped.contiguous())
        # conv_cap_4_1 = x
        conv_cap_4_1 = self.encoder_conv_caps[1](x)

        shape = conv_cap_4_1.size()
        conv_cap_4_1 = conv_cap_4_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        conv_cap_4_1 = self.efficientViT3Dblock_bottleneck(conv_cap_4_1)

        # Downsampled predictions
        norm = torch.linalg.norm(conv_cap_4_1, dim=2)

        # Expanding
        if self.connection == "skip":
            # ###########################################################################################
            # x = self.patchExpandingblock_3(conv_cap_4_1)
            # x = torch.cat((x, conv_3_1), dim=1)
            # x = self.efficientViT3Dblock_decoder_3(x)
            # x = self.relu(x)
            # x = self.patchExpandingblock_2(x)
            # x = torch.cat((x, conv_2_1), dim=1)
            # x = self.efficientViT3Dblock_decoder_2(x)
            # x = self.relu(x)
            # x = self.patchExpandingblock_1(x)
            # x = torch.cat((x, conv_1_1), dim=1)
            # ###########################################################################################
            x = self.decoder_conv[0](conv_cap_4_1)
            x = torch.cat((x, conv_3_1), dim=1)
            x = self.decoder_conv[1](x)
            x = self.efficientViT3Dblock_decoder_3(x)
            x = self.decoder_conv[2](x)
            x = torch.cat((x, conv_2_1), dim=1)
            x = self.decoder_conv[3](x)
            x = self.efficientViT3Dblock_decoder_2(x)
            x = self.decoder_conv[4](x)
            x = torch.cat((x, conv_1_1), dim=1)

            # extend decover and skip connection
            # x = self.add_deconvs[0](x)
            # x = torch.cat((x, fe_1), dim=1)
            # x = self.add_deconvs[1](x)
            # x = torch.cat((x, fe_0), dim=1)

        logits = self.decoder_conv[5](x)

        # Reconstructing
        reconstructions = self.reconstruct_branch(x)

        # Calculating losses
        loss, cls_loss, rec_loss = self.losses(images, labels, norm, logits, reconstructions)

        self.log("margin_loss", cls_loss[0], on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{self.cls_loss}_loss", cls_loss[1], on_step=False, on_epoch=True, sync_dist=True)
        self.log("reconstruction_loss", rec_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        val_outputs = sliding_window_inference(
            images,
            roi_size=self.val_patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.forward,
            overlap=self.overlap,
        )

        val_outputs = [self.post_pred(val_output) for val_output in decollate_batch(val_outputs)]
        labels = [self.post_label(label) for label in decollate_batch(labels)]
        self.dice_metric(y_pred=val_outputs, y=labels)
        self.precision_metric(y_pred=val_outputs, y=labels)
        self.recall_metric(y_pred=val_outputs, y=labels)

    def validation_epoch_end(self, outputs):
        dice_scores = self.dice_metric.aggregate()
        mean_val_dice = torch.mean(dice_scores)
        print(f"mean_val_dice:{mean_val_dice}")
        self.log("val_dice", mean_val_dice, sync_dist=True)
        for i, dice_score in enumerate(dice_scores):
            self.log(f"val_dice_class {i + 1}", dice_score, sync_dist=True)
            print(f"val_dice_class {i + 1}:{dice_score}")
        self.dice_metric.reset()

        precisions = self.precision_metric.aggregate()[0]
        mean_val_precision = torch.mean(precisions)
        print(f"mean_val_precision:{mean_val_precision}")
        for i, precision in enumerate(precisions):
            print(f"val_precision_class {i + 1}:{precision}")
        self.precision_metric.reset()

        recalls = self.recall_metric.aggregate()[0]
        mean_val_recall = torch.mean(recalls)
        print(f"mean_val_recall:{mean_val_recall}")
        for i, recall in enumerate(recalls):
            print(f"val_recall_class {i + 1}:{recall}")
        self.recall_metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images = batch["image"]
        outputs = sliding_window_inference(
            images,
            roi_size=self.val_patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.forward,
            overlap=self.overlap,
        )
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1, patience=5),
            "monitor": "val_dice",
            "frequency": self.val_frequency,
        }
        '''
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2),
            "monitor": "val_dice",
            "frequency": self.val_frequency
        }
        '''
        return [optimizer], [scheduler]

    def losses(self, volumes, labels, norm, pred, reconstructions):
        mask = torch.gt(labels, 0)
        rec_loss = torch.sum(self.reconstruction_loss(volumes * mask, reconstructions * mask), dim=(1, 2, 3, 4)) / (
                torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        )
        rec_loss = torch.mean(rec_loss)

        downsample_labels = F.interpolate(
            one_hot(labels, self.out_channels), scale_factor=0.125, mode="trilinear", align_corners=False
        )
        cls_loss1 = self.classification_loss1(norm, downsample_labels)
        cls_loss2 = self.classification_loss2(pred, labels)

        return (
            self.margin_loss_weight * cls_loss1 + cls_loss2 + self.rec_loss_weight * rec_loss,
            [cls_loss1, cls_loss2],
            rec_loss,
        )

    def _build_encoder(self):
        # self.encoder_convs = nn.ModuleList()
        # self.encoder_convs.append(
        #     nn.Conv3d(64, 128, 3, stride=2, padding=1)
        # )
        # # self.encoder_convs.append(
        # #     nn.Conv3d(128, 128, 5, stride=1, padding=2)
        # # )
        # self.encoder_convs.append(
        #     nn.Conv3d(128, 128, 3, stride=2, padding=1)
        # )
        # # self.encoder_convs.append(
        # #     nn.Conv3d(128, 128, 5, stride=1, padding=2)
        # # )
        #
        # for i in range(len(self.encoder_convs)):
        #     torch.nn.init.normal_(self.encoder_convs[i].weight, std=0.1)

        self.relu = nn.ReLU(inplace=True)

        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 3

        self.encoder_conv_caps.append(
            ConvSlimCapsule3D(
                kernel_size=self.encoder_kernel_size,
                input_dim=8,
                output_dim=8,
                input_atoms=16,
                output_atoms=32,
                stride=2,
                padding=1,
                dilation=1,
                num_routing=3,
                share_weight=self.share_weight,
            )
        )

        self.encoder_conv_caps.append(
            ConvSlimCapsule3D(
                kernel_size=self.encoder_kernel_size,
                input_dim=8,
                output_dim=self.out_channels,
                input_atoms=32,
                output_atoms=64,
                stride=1,
                padding=1,
                dilation=1,
                num_routing=3,
                share_weight=self.share_weight,
            )
        )


    def _build_decoder(self):
        # self.add_deconvs = nn.ModuleList()
        # self.add_deconvs.append(
        #     nn.ConvTranspose3d(128, 32, 1, 1)
        # )
        # self.add_deconvs.append(
        #     nn.ConvTranspose3d(64, 16, 1, 1)
        # )
        self.decoder_conv = nn.ModuleList()
        if self.connection == "skip":
            self.decoder_in_channels = [self.out_channels * 64, 384, 128, 256, 64, 128]
            self.decoder_out_channels = [256, 128, 128, 64, 64, self.out_channels]

        for i in range(6):
            if i == 5:
                self.decoder_conv.append(
                    Conv["conv", 3](self.decoder_in_channels[i], self.decoder_out_channels[i], kernel_size=1)
                )

            elif i % 2 == 0:
                self.decoder_conv.append(
                    UpSample(
                        dimensions=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        scale_factor=2,
                    )
                )
            else:
                self.decoder_conv.append(
                    Convolution(
                        dimensions=3,
                        kernel_size=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        strides=1,
                        padding=1,
                        bias=False,
                    )
                )

    def _build_reconstruct_branch(self):
        self.reconstruct_branch = nn.Sequential(
            nn.Conv3d(self.decoder_in_channels[-1], 64, 1),
            # nn.Conv3d(32, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, self.in_channels, 1),
            nn.Sigmoid(),
        )

    def _build_3D_EfficientViTBlock(self):
        # self.efficientViT3Dblock_encoder_1 = EfficientViTBlock3D(type='s', ed=64, kd=16, ar=1, resolution=self.val_patch_size[0])
        self.efficientViT3Dblock_encoder_2 = EfficientViTBlock3D(type='s', ed=128, kd=16, ar=2, resolution=self.val_patch_size[0]//2)
        self.efficientViT3Dblock_encoder_3 = EfficientViTBlock3D(type='s', ed=128, kd=16, ar=2, resolution=self.val_patch_size[0]//4)

        # self.efficientViT3Dblock_decoder_1 = EfficientViTBlock3D(type='s', ed=128, kd=16, ar=2, resolution=self.val_patch_size[0])
        self.efficientViT3Dblock_decoder_2 = EfficientViTBlock3D(type='s', ed=64, kd=16, ar=1, resolution=self.val_patch_size[0]//2)
        self.efficientViT3Dblock_decoder_3 = EfficientViTBlock3D(type='s', ed=128, kd=16, ar=2, resolution=self.val_patch_size[0]//4)

        self.efficientViT3Dblock_bottleneck = EfficientViTBlock3D(type='s', ed=self.out_channels * 64, kd=16,
                                                                  ar=self.out_channels, resolution=self.val_patch_size[0]//8)

        self.patchMergingblock_1 = PatchMerging3D(input_dim=64, output_dim=128)
        self.patchMergingblock_2 = PatchMerging3D(input_dim=128, output_dim=128)

        # self.patchExpandingblock_1 = PatchExpand3D(input_dim=256, output_dim=64)
        # self.patchExpandingblock_2 = PatchExpand3D(input_dim=256, output_dim=128)
        # self.patchExpandingblock_3 = PatchExpand3D(input_dim=self.out_channels * 64, output_dim=128)
