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

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import pdb
from monai.data import decollate_batch
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
from kornia.filters import SpatialGradient3d
from monai.networks import one_hot
from monai import transforms

from monai.losses import DiceCELoss

def create_slice_output(image_np, pred_roi_np_cnn, actual_roi_np, output_directory, subject,argmax=False):

    print("Creating image for :", subject)
    num_slices = actual_roi_np.shape[-1]
    num_images = (num_slices // 64)
    print(np.unique(pred_roi_np_cnn))
    print(np.unique(actual_roi_np))
    for img_num in range(num_images):
        plt.figure(img_num, figsize=(12,6*64))
        for z in range(64):
            z_index = 64*img_num + z 
            plt.subplot(64,3, 3*z + 1)
            # plt.imshow(np.rot90(image_np[:, :, z_index]), cmap = 'gray')
            plt.imshow(image_np[:, :, z_index], cmap = 'gray')
            plt.axis('off')
            plt.title('Image: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

            plt.subplot(64,3, 3*z + 2)
            # plt.imshow(np.rot90(actual_roi_np[:, :, z_index]))
            plt.imshow(actual_roi_np[:,:,z_index].T)
            # else:
                # plt.imshow(actual_roi_np[:,:,:,z_index].T)
            # plt.colorbar()
            plt.axis('off')
            plt.title('Label: z= ' + str(z_index + 1) + '/' + str(actual_roi_np.shape[-1]))

            plt.subplot(64,3, 3*z + 3)
            # plt.imshow(np.rot90(pred_roi_np_cnn[ :, :, z_index]))
            if argmax:
                plt.imshow(pred_roi_np_cnn[:, :, z_index].T)
            else:
                plt.imshow(pred_roi_np_cnn[:,:, :, z_index].T)
            # plt.colorbar()
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

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, edge_loss_func, args, lambda1=1.0, lambda2=0.5, lambda3=0.1,post_label=None, boundary_loss_func=None):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    spatial_gradient = SpatialGradient3d()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            
            '''
            attention_logits = model.get_attention_logits()
            logit_list = model.get_intermediate_attention_logits()

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

            #For main shape stream output
            boundary_loss = edge_loss_func(attention_logits, attention_target)
            
            #For intermediate layers -- Deep Supervision
            scale_weight = 0.5
            for scaled_attention_logit in logit_list:
                scaled_attention_target = torch.nn.functional.interpolate(attention_target, scaled_attention_logit.shape[2:])
                boundary_loss += scale_weight * edge_loss_func(scaled_attention_logit, scaled_attention_target)
                scale_weight /=2
            '''
            seg_loss = loss_func(logits, target)
            loss = seg_loss
            # loss = seg_loss +  boundary_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        # if args.rank == 0:
            # print(
            #     "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            #     "loss: {:.4f}".format(run_loss.avg),
            #     "time {:.2f}s".format(time.time() - start_time),
            # )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, acc_func_classwise, args, model_inferer=None, post_label=None, post_pred=None, create_images=False):
    model.eval()
    run_acc = AverageMeter()
    run_acc_classwise = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            # if create_images:
            #     img_name = batch_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            #     create_slice_output(data.cpu().numpy()[0][0], np.argmax(val_output_convert[0].cpu().numpy(), axis=0).astype(np.uint8), np.argmax(val_labels_convert[0].cpu().numpy(), axis=0).astype(np.uint8),args.output_dir, img_name )
            acc_func.reset()
            acc_func_classwise.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc_func_classwise(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc_classwise, not_nans_classwise = acc_func_classwise.aggregate()
            acc = acc.cuda(args.rank)
            acc_classwise = acc_classwise.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                run_acc_classwise.update(acc_classwise.cpu().numpy(), n=not_nans_classwise.cpu().numpy())

            if args.rank == 0:
                panc_acc = run_acc_classwise.avg[0]
                tumor_acc = run_acc_classwise.avg[1]
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    " Pancreas : ",
                    panc_acc,
                    " Tumor : ",
                    tumor_acc,
                    " | Average : ",
                    avg_acc,
                    " time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return (run_acc.avg, run_acc_classwise.avg)


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    edge_loss_func,
    acc_func,
    acc_func_classwise,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
    lambda1=1.0,
    lambda2=0.5,
    lambda3=0.1
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(time.ctime(), "Epoch {}/{}".format(epoch, args.max_epochs))
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, edge_loss_func=edge_loss_func, args=args, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, post_label=post_label
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc, val_avg_acc_classwise = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                acc_func_classwise=acc_func_classwise,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
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
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
