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

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.networks.blocks import  UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from torchsummary import summary
from monai.networks.blocks import Convolution
from monai.networks.blocks import SimpleASPP
from collections.abc import Sequence
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.networks.nets.swin_unetr import SwinTransformer
from kornia.filters import SpatialGradient3d


parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained_weights.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=600, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=2e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=50, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=24, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--replace_ll_temp", default=3, type=int, help="to replace the number of channels in last layer to load pt weights")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-21.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=189.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--lambda1",default=1.0,type=float)
parser.add_argument("--lambda2",default=0.5,type=float)
parser.add_argument("--lambda3",default=0.1,type=float)



class ShapeSwinUNETRV8(SwinUNETR):
    def __init__(self, img_size,spatial_dims, in_channels, out_channels, feature_size, drop_rate, attn_drop_rate, dropout_path_rate, use_checkpoint, norm_name):
        super(ShapeSwinUNETRV8, self).__init__(img_size=img_size, in_channels=in_channels, out_channels=out_channels, feature_size=feature_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, dropout_path_rate=dropout_path_rate, use_checkpoint=use_checkpoint)

        #Attempting to use hidden_state_output[1,2,3,4] as inputs for attention layer / shape stream
        #To try and reduce the number of params => option 1 (downsampling by stride 2 but keeping num filters = second input size (192,384,768) same as in_channels throughout)
        #Res blocks in this case

        #[4,48,48,48,48] => [4,48,48,48,48] => [4,48,48,48,48]
        self.hs0_conv1 = UnetrBasicBlock(
            spatial_dims,
            feature_size,
            feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,48,48,48,48] => [4,48,48,48,48] => [4,96,24,24,24]
        self.hs0_conv2 = UnetrBasicBlock(
            spatial_dims,
            feature_size,
            2*feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True
        )

        #[4,96,24,24,24] => [4,96,24,24,24] => [4,96,24,24,24]
        self.hs1_conv1 = UnetrBasicBlock(
            spatial_dims,
            2*feature_size,
            2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )
        #[4,96,24,24,24] => [4,96,24,24,24] => [4,96,24,24,24]
        self.hs1_conv2 = UnetrBasicBlock(
            spatial_dims,
            2*feature_size,
            2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,192,12,12,12] => [4,192,12,12,12] => [4,192,12,12,12]
        self.hs2_conv1 = UnetrBasicBlock(
            spatial_dims,
            4*feature_size,
            4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,192,12,12,12] => [4,192,12,12,12] => [4,192,12,12,12]
        self.hs2_conv2 = UnetrBasicBlock(
            spatial_dims,
            4*feature_size,
            4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,384,6,6,6] => [4,384,6,6,6] => [4,384,6,6,6]
        self.hs3_conv1 = UnetrBasicBlock(
            spatial_dims,
            8*feature_size,
            8*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,384,6,6,6] => [4,384,6,6,6] => [4,384,6,6,6]
        self.hs3_conv2 = UnetrBasicBlock(
            spatial_dims,
            8*feature_size,
            8*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,768,3,3,3] => [4,768,3,3,3] => [4,768,3,3,3]
        self.hs4_conv1 = UnetrBasicBlock(
            spatial_dims,
            16*feature_size,
            16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,768,3,3,3] => [4,768,3,3,3] => [4,768,3,3,3]
        self.hs4_conv2 = UnetrBasicBlock(
            spatial_dims,
            16*feature_size,
            16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        self.attn0_conv1_1 = get_conv_layer(
            spatial_dims,
            4*feature_size,
            2*feature_size,
            kernel_size=1,
            stride=1,
            dropout=None,
            act=None,
            norm=norm_name,
            conv_only=False,
        )

        # 1x1 conv layers for the attention blocks
        self.attn1_conv1_1 = get_conv_layer(
            spatial_dims,
            8*feature_size,
            4*feature_size,
            kernel_size=1,
            stride=1,
            dropout=None,
            act=None,
            norm=norm_name,
            conv_only=False,
        )

        self.attn2_conv1_1 = get_conv_layer(
            spatial_dims,
            16*feature_size,
            8*feature_size,
            kernel_size=1,
            stride=1,
            dropout=None,
            act=None,
            norm=norm_name,
            conv_only=False,
        )

        self.attn3_conv1_1 = get_conv_layer(
            spatial_dims,
            32*feature_size,
            16*feature_size,
            kernel_size=1,
            stride=1,
            dropout=None,
            act=None,
            norm=norm_name,
            conv_only=False,
        )

        # No shape change conv blocks to pass attention layers through
        self.transition0 = UnetrBasicBlock(
            spatial_dims,
            2*feature_size,
            4*feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True
        )

        self.transition1= UnetrBasicBlock(
            spatial_dims,
            4*feature_size,
            8*feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True
        )

        self.transition2= UnetrBasicBlock(
            spatial_dims,
            8*feature_size,
            16*feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True
        )

        self.ASPP = SimpleASPP(
            spatial_dims=spatial_dims,
            in_channels = 32*feature_size, 
            conv_out_channels = 4*feature_size #*4 from len(kernels) for ASPP =16, see documentation
        )

        #[4,768,3,3,3]=>[4,384,6,6,6]
        self.upsample_attn_tpconv1 = get_conv_layer(
            spatial_dims,
            16*feature_size,
            8*feature_size,
            kernel_size=2,
            stride=2,
            conv_only=False,
            is_transposed=True,
            norm=norm_name,
        )
        self.upsample_attn_conv1 = UnetrBasicBlock(
            spatial_dims,
            8*feature_size,
            8*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,384,6,6,6]=>[4,192,12,12,12]
        self.upsample_attn_tpconv2 = get_conv_layer(
            spatial_dims,
            8*feature_size,
            4*feature_size,
            kernel_size=2,
            stride=2,
            conv_only=False,
            norm=norm_name,
            is_transposed=True,
        )

        self.upsample_attn_conv2 = UnetrBasicBlock(
            spatial_dims,
            4*feature_size,
            4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,192,12,12,12]=>[4,96,24,24,24]
        self.upsample_attn_tpconv3 = get_conv_layer(
            spatial_dims,
            4*feature_size,
            2*feature_size,
            kernel_size=2,
            stride=2,
            conv_only=False,
            norm=norm_name,
            is_transposed=True
        )

        self.upsample_attn_conv3 = UnetrBasicBlock(
            spatial_dims,
            2*feature_size,
            2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,96,24,24,24]=>[4,48,48,48,48]
        self.upsample_attn_tpconv4 = get_conv_layer(
            spatial_dims,
            2*feature_size,
            feature_size,
            kernel_size=2,
            stride=2,
            conv_only=False,
            norm=norm_name,
            is_transposed=True
        )

        self.upsample_attn_conv4 = UnetrBasicBlock(
            spatial_dims,
            feature_size,
            feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        #[4,48,48,48,48]=>[4,48,96,96,96]
        self.upsample_attn_tpconv5 = get_conv_layer(
            spatial_dims,
            feature_size,
            feature_size,
            kernel_size=2,
            stride=2,
            conv_only=False,
            norm=norm_name,
            is_transposed=True,
        )

        self.upsample_attn_conv5 = UnetrBasicBlock(
            spatial_dims,
            feature_size,
            feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        self.sigmoid = get_act_layer(name="sigmoid")
        self.attn_out_layer = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.attn0_out_layer = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.attn1_out_layer = UnetOutBlock(spatial_dims=spatial_dims, in_channels=2*feature_size, out_channels=out_channels)
        self.attn2_out_layer = UnetOutBlock(spatial_dims=spatial_dims, in_channels=4*feature_size, out_channels=out_channels)
        self.attn3_out_layer = UnetOutBlock(spatial_dims=spatial_dims, in_channels=8*feature_size, out_channels=out_channels)

        #Note: Batch Size hardcoded for attention_logits here
        self.attention_logits = torch.zeros(4,feature_size,img_size[0],img_size[1],img_size[2])

        
    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        ##SHAPE STREAM PART

        #attention layer 0
        s0 = self.hs0_conv1(hidden_states_out[0])
        s0 = self.hs0_conv2(s0)
        m0 = self.hs1_conv1(hidden_states_out[1])
        m0 = self.hs1_conv2(m0)
        alpha0 = torch.cat((s0, m0), dim=1)
        alpha0 = self.attn0_conv1_1(alpha0)
        alpha0 = self.sigmoid(alpha0)
        attn0 = s0 * alpha0
        self.attn0 = attn0

        #attention layer 1
        attn0 = self.transition0(attn0)
        m1 = self.hs2_conv1(hidden_states_out[2])
        m1 = self.hs2_conv2(m1)
        alpha1 = torch.cat((attn0, m1), dim=1)
        alpha1 = self.attn1_conv1_1(alpha1)
        alpha1 = self.sigmoid(alpha1)
        attn1 = attn0 * alpha1
        self.attn1 = attn1

        #attention layer 2
        attn1 = self.transition1(attn1)
        m2 = self.hs3_conv1(hidden_states_out[3])
        m2 = self.hs3_conv2(m2)
        alpha2 = torch.cat((attn1, m2), dim=1)
        alpha2 = self.attn2_conv1_1(alpha2)
        alpha2 = self.sigmoid(alpha2)
        attn2 = attn1 * alpha2
        self.attn2 = attn2

        #attention layer 3
        attn2 = self.transition2(attn2)
        m3 = self.hs4_conv1(hidden_states_out[4])
        m3 = self.hs4_conv2(m3)
        alpha3 = torch.cat((attn2, m3), dim=1)
        alpha3 = self.attn3_conv1_1(alpha3)
        alpha3 = self.sigmoid(alpha3)
        attn3 = attn2 * alpha3
        self.attn3 = attn3
        
        #Upsampling to get attention logits from attn3 output
        attn_out = self.upsample_attn_tpconv1(attn3)
        attn_out = self.upsample_attn_conv1(attn_out)
        self.attn3_out = self.attn3_out_layer(attn_out)
        
        attn_out = self.upsample_attn_tpconv2(attn_out)
        attn_out = self.upsample_attn_conv2(attn_out)
        self.attn2_out = self.attn2_out_layer(attn_out)
        
        attn_out = self.upsample_attn_tpconv3(attn_out)
        attn_out = self.upsample_attn_conv3(attn_out)
        self.attn1_out = self.attn1_out_layer(attn_out)
        
        attn_out = self.upsample_attn_tpconv4(attn_out)
        attn_out = self.upsample_attn_conv4(attn_out)
        self.attn0_out = self.attn0_out_layer(attn_out)
        
        attn_out = self.upsample_attn_tpconv5(attn_out)
        attn_out = self.upsample_attn_conv5(attn_out)
        self.attention_logits = self.attn_out_layer(attn_out)
        
        ##END OF SHAPE STREAM PART

        enc_final = torch.cat((dec4, attn3), dim=1)
        enc_final = self.ASPP(enc_final)

        dec3 = self.decoder5(enc_final, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
    
        return logits
    
    def get_attention_logits(self):
        return self.attention_logits
    
    def get_intermediate_attention_logits(self):
        return (self.attn0_out, self.attn1_out, self.attn2_out, self.attn3_out)

def get_beta_values(train_loader, args):
    spatial_gradient = SpatialGradient3d()
    overall_beta_counts = [0.0,0.0,0.0]
    overall_total = 0.0
    for idx, batch_data in enumerate(train_loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        attention_target1 = target.clone()
        attention_target1[attention_target1==2] = 1 #set tumor to pancreas, this is now only pancreas target
        attention_target1 = torch.abs(spatial_gradient(attention_target1))
        attention_target1 = torch.max(attention_target1, axis=1)[0]

        attention_target2 = target.clone()
        attention_target2[attention_target2==1]= 0 #set pancreas to background, this is only tumor target
        attention_target2[attention_target2==2]= 1
        attention_target2 = torch.abs(spatial_gradient(attention_target2))
        attention_target2 = torch.max(attention_target2, axis=1)[0]

        # Creating [batch,1,sp_x, sp_y,sp_z] shapped attention_target , with 0 for background, 1 for pancreas edge, and 2 for tumor edge
        attention_target = torch.zeros_like(target)
        pancreas_edges = torch.argmax(attention_target1,axis=1,keepdim=True)
        tumor_edges = torch.argmax(attention_target2,axis=1,keepdim=True)
        attention_target[pancreas_edges>0]=1
        attention_target[tumor_edges>0]=2 

        bin_counts = torch.unique(attention_target, return_counts=True)[1]
        if (bin_counts.size(dim=0)==1):
            bin_counts = torch.cat((bin_counts,torch.Tensor([0]).cuda(0)))
        if (bin_counts.size(dim=0)==2):
            bin_counts = torch.cat((bin_counts,torch.Tensor([0]).cuda(0)))
        
        total = torch.numel(attention_target)
        
        overall_beta_counts[0] += bin_counts[0]
        overall_beta_counts[1] += bin_counts[1]
        overall_beta_counts[2] += bin_counts[2]
        overall_total += total 
    
    beta_1 = (overall_total-overall_beta_counts[0])/overall_total
    beta_2 = (overall_total-overall_beta_counts[1])/overall_total
    beta_3 = (overall_total-overall_beta_counts[2])/overall_total
    print ("Beta_values: ", str((beta_1, beta_2, beta_3)))
    return (beta_1, beta_2, beta_3)



def main():
    args = parser.parse_args()
    # torch.cuda.set_per_process_memory_fraction(0.7, 0)
    print(torch.cuda.mem_get_info(device=0))
    # torch.cuda.empty_cache()
    print(args)
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=args.rank, args=args)

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir

    # model = ShapeSwinUNETRV8(
    #     img_size=(args.roi_x, args.roi_y, args.roi_z),
    #     spatial_dims = args.spatial_dims,
    #     in_channels=args.in_channels,
    #     out_channels=args.replace_ll_temp, #to load pretrained weights
    #     feature_size=args.feature_size,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     dropout_path_rate=args.dropout_path_rate,
    #     use_checkpoint=args.use_checkpoint,
    #     norm_name="instance"
    # )

    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.replace_ll_temp, #to load pretrained weights
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint    
    )

    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")
        model.out = UnetOutBlock(spatial_dims=args.spatial_dims, in_channels=args.feature_size, out_channels=args.out_channels)
    
    if args.use_ssl_pretrained:
        try:
            model_dict = torch.load("./pretrained_models/model_swinvit.pt")
            state_dict = model_dict["state_dict"]
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
                
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised Swin UNETR backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    
    beta_1, beta_2, beta_3 = get_beta_values(loader[0], args)
    edge_loss_func = DiceCELoss(to_onehot_y=True, softmax=True,  ce_weight=torch.Tensor([beta_1, beta_2, beta_3]).cuda(0), lambda_dice = args.lambda2, lambda_ce = args.lambda3)

    post_label = AsDiscrete(to_onehot=3, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=3, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    dice_acc_classwise = DiceMetric(include_background=False, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        edge_loss_func=edge_loss_func,
        acc_func=dice_acc,
        acc_func_classwise=dice_acc_classwise,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3
    )
    return accuracy


if __name__ == "__main__":
    main()
