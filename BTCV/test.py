# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
from monai.utils.enums import MetricReduction


import nibabel as nib
import numpy as np
import numpy.ma as ma
import torch
from utils.data_utils import get_loader
from utils.utils import dice, resample_3d
# import matplotlib.transforms as mtransforms

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import matplotlib.pyplot as plt
import pandas as pd
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from trainer import val_epoch

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="model.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.7, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-87.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=199.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.5, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.25, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.5, type=float, help="RandScaleIntensityd aug probability")
# parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")


def create_slice_output(image_np, pred_roi_np_cnn, actual_roi_np, output_directory, subject):

    num_slices = actual_roi_np.shape[-1]
    num_images = (num_slices // 64)

    for img_num in range(num_images):
        plt.figure(img_num, figsize=(12,6*64))
        for z in range(64):
            z_index = 64*img_num + z 
            plt.subplot(64,3, 3*z + 1)
            plt.imshow(np.rot90(image_np[:, :, z_index]), cmap = 'gray')
            plt.axis('off')
            plt.title('Image: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

            plt.subplot(64,3, 3*z + 2)
            plt.imshow(np.rot90(actual_roi_np[:, :, z_index]))
            plt.axis('off')
            plt.title('Label: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

            plt.subplot(64,3, 3*z + 3)
            plt.imshow(np.rot90(pred_roi_np_cnn[ :, :, z_index]))
            plt.axis('off')
            plt.title('Pred: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

        plt.savefig(os.path.join(output_directory, subject+'_'+str(img_num)+".png"), bbox_inches="tight")
        plt.figure(img_num).clear()

    #For the last few slices 
    plt.figure(num_images, figsize=(12,6*64))
    z = 0
    img_num = num_images
    for k in range(num_images * 64, num_slices):
        z_index = 64*img_num + z 

        plt.subplot(64,3, 3*z + 1 )
        plt.imshow(np.rot90(image_np[:, :, z_index]), cmap = 'gray')
        plt.axis('off')
        plt.title('Image: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

        plt.subplot(64,3, 3*z + 2)
        plt.imshow(np.rot90(actual_roi_np[:, :, z_index]))
        plt.axis('off')
        plt.title('Label: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

        plt.subplot(64,3, 3*z + 3)
        plt.imshow(np.rot90(pred_roi_np_cnn[:, :, z_index]))
        plt.axis('off')
        plt.title('Pred: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

        z+=1
    plt.savefig(os.path.join(output_directory, subject+'_'+str(num_images)+".png"))
    plt.figure(num_images).clear()


def main():
    args = parser.parse_args()
    args.test_mode = True
    args.rank=0
    args.amp=True
    args.max_epochs=1
    output_directory = "./outputs_runv5/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=4,
        predictor=model,
        overlap=args.infer_overlap,
    )
    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    dice_acc_classwise = DiceMetric(include_background=False, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    args.output_dir = output_directory
    val_avg_acc, val_avg_acc_classwise = val_epoch(
        model,
        val_loader,
        epoch=1,
        acc_func=dice_acc,
        acc_func_classwise=dice_acc_classwise,
        model_inferer=model_inferer,
        args=args,
        post_label=post_label,
        post_pred=post_pred,
        create_images=True
    )

    val_avg_acc = np.mean(val_avg_acc)

    if args.rank == 0:
        print(
            "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
            "average accuracy : ",
            val_avg_acc,
            "pancreas : ",
            val_avg_acc_classwise[0],
            "tumor : ",
            val_avg_acc_classwise[1],
            "time {:.2f}s".format(time.time() - epoch_time),
        )        

    # with torch.no_grad():
    #     dice_list_case = []
    #     for i, batch in enumerate(val_loader):
    #         val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
    #         # print(val_inputs.shape, val_labels.shape)
    #         original_affine = batch["label_meta_dict"]["affine"][0].numpy()
    #         _, _, h, w, d = val_labels.shape
    #         target_shape = (h, w, d)
    #         img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
    #         print("Inference on case {}".format(img_name))
    #         val_outputs = sliding_window_inference(
    #             val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
    #         )
    #         val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
    #         # print(val_outputs.shape)
    #         val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
    #         # print(val_outputs.shape, val_labels.shape)
    #         val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
    #         # print(np.unique(val_labels))
    #         val_outputs = resample_3d(val_outputs, target_shape)
    #         # print(val_outputs.shape, val_labels.shape)
    #         dice_list_sub = []
    #         create_slice_output(val_inputs.cpu().numpy()[0, 0, :, :, :], val_outputs, val_labels, output_directory, img_name) #Create images for outputs
    #         for i in range(1, 3): #Ignore the background
    #             organ_Dice = dice(val_outputs == i, val_labels == i)
    #             dice_list_sub.append(organ_Dice)
    #         # mean_dice = np.mean(dice_list_sub)
    #         print("Pancreas: {} | Tumor: {} | Mean Organ Dice: {}".format(dice_list_sub[0], dice_list_sub[1], np.mean(dice_list_sub)))
    #         dice_list_case.append(dice_list_sub)
    #         mean_dice = np.mean(dice_list_case, axis=0)
    #         # print(mean_dice)
    #         # print(mean_dice[0], mean_dice[1])
    #         # nib.save(
    #         #     nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
    #         # )
    #     # print (dice_list_case)
    #     mean_dice = np.mean(dice_list_case, axis =0)
    #     print("Overall Pancreas: {} | Overall Tumor: {} | Overall Mean: {}".format(mean_dice[0], mean_dice[1], np.mean(mean_dice)))


if __name__ == "__main__":
    main()


