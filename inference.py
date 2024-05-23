import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
           
import numpy as np
import torch
from datamodule.iseg import ISeg2017DataModule
from module.effiViTcaps import EffiViTCaps3D
from monai.data import NiftiSaver, decollate_batch
from monai.metrics import ConfusionMatrixMetric, DiceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType, MapLabelValue
from monai.utils import set_determinism
from pytorch_lightning import Trainer
from tqdm import tqdm

def inference () :

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="C:/Users/pc/Desktop/3D-EffiViTCaps")
    parser.add_argument("--save_image", type=int, default=1, help="Save image or not")
    parser = Trainer.add_argparse_args(parser)

    val_parser = parser.add_argument_group("Validation config")
    val_parser.add_argument("--output_dir", type=str, default="C:/Users/pc/Desktop/3D-EffiViTCaps/output_dir")
    val_parser.add_argument("--model_name", type=str, default="modified-ucaps", help="ucaps / segcaps-2d / segcaps-3d / unet")
    val_parser.add_argument(
        "--dataset", type=str, default="iseg2017", help="iseg2017 / task02_heart / task04_hippocampus / luna16"
    )
    val_parser.add_argument("--fold", type=int, default=4)
    val_parser.add_argument(
        "--checkpoint_path", type=str, default="C:/Users/pc/Desktop/3D-EffiViTCaps/weights/3D-EffiViTCaps_iseg2017.ckpt", help='/path/to/trained_model. Set to "" for none.'
    )

    temp_args, _ = parser.parse_known_args()

    parser, model_parser = EffiViTCaps3D.add_model_specific_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    print("Validation config:")
    for a in val_parser._group_actions:
        print("\t{}:\t{}".format(a.dest, dict_args[a.dest]))

    set_determinism(seed=0)
    
    data_module = ISeg2017DataModule(
        **dict_args,
    )
    map_label = MapLabelValue(target_labels=[0, 10, 150, 250], orig_labels=[0, 1, 2, 3], dtype=np.float32)

    data_module.setup("validate")
    val_loader = data_module.val_dataloader()
    val_batch_size = 1

    net = EffiViTCaps3D.load_from_checkpoint(
        args.checkpoint_path,
        val_patch_size=args.val_patch_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
    )
    print("Load trained model!!!")

    # Prediction

    trainer = Trainer.from_argparse_args(args, accelerator='gpu')
    outputs = trainer.predict(net, dataloaders=val_loader)

    print (outputs[0].shape, len (outputs))

    # Calculate metric and visualize

    n_classes = net.out_channels
    save_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=False, n_classes=n_classes)])

    pred_saver = NiftiSaver(
        output_dir=args.output_dir,
        output_postfix=f"{args.model_name}_prediction",
        resample=False,
        data_root_dir=args.root_dir,
        output_dtype=np.uint8,
    )

    for i, data in enumerate(tqdm(val_loader)):
        labels = data["label"]

        val_outputs = outputs[i].cpu()

        if args.dataset == "iseg2017":
            pred_saver.save_batch(
                map_label(torch.stack([save_pred(i) for i in decollate_batch(val_outputs)]).cpu()),
                meta_data={
                    "filename_or_obj": data["label_meta_dict"]["filename_or_obj"],
                    "original_affine": data["label_meta_dict"]["original_affine"],
                    "affine": data["label_meta_dict"]["affine"],
                },
            )