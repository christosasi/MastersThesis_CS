#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt

import time
import torch
import functools
import datetime as dt

import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
import kornia
from kornia.augmentation.container import AugmentationSequential
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F # Make sure to import functional

from torchvision import transforms
import nibabel as nib
from models import BrainNetwork
from models import *
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
sys.path.append('autoencoder2/')
from autoencoder2.convnext import ConvnextXL
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
# custom functions #
import utils
import pickle
from diffusers import AutoencoderKL    
import logging as logging
import types
import wandb

from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='bf16')
device = accelerator.device
accelerator.print(f"Using device: {device}")


rank = accelerator.process_index
world_size = accelerator.num_processes


per_device_batch_size = 1  # or adjust based on your experiments


num_devices = torch.cuda.device_count()
num_devices = 2
num_workers = 2





n_blocks=4
hidden_dim = 1024
#set data type to bfloat16 for mixed precision training 
data_type = torch.bfloat16
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True



# Store the original method reference

# Paths
voxel_data_path = #define input_dirs
outlier_mask_path = #define input_dirs
masked_movie_data = np.load(f"input_dirs")
frames_data_path = '#input_dirs'
data_path = "input_dirs"
cache_dir = "cache_dir"
output_dir = #define output_dir


#load the frames data
all_pulses = list(range(7, 1007))  # This gives 7, 8, ..., 1006 (1000 numbers)
# Explicitly set exact splits
train_pulses = all_pulses[7:707]   # First 700 for training
val_pulses   = all_pulses[707:907]  # Next 200 for validation
test_pulses  = all_pulses[907:1007]     # Remaining 100 for testing



# #load the frames data
# all_pulses = list(range(7, 107))  # This gives 7, 8, ..., 1006 (1000 numbers)
# # Explicitly set exact splits
# train_pulses = all_pulses[7:77]   # First 700 for training
# val_pulses   = all_pulses[77:97]  # Next 200 for validation
# test_pulses  = all_pulses[97:107]     # Remaining 100 for testing




transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

class MovieBetasDataset(Dataset):
    """
    Each sample = (center PNG frame, masked betas) for a given pulse.
    - masked_data: shape (9423, Npulses)
    - frames_root: folder with subfolders pulse1..pulseN
    - pulse_list: which pulse IDs to include (e.g. [1,2,3,...,70])
    - transform: optional image transform (e.g., Resize, ToTensor, Normalize)
    """
    def __init__(self, masked_data, frames_root, pulse_list, transform=None):
        super().__init__()
        self.masked_data = masked_data
        self.frames_root = frames_root
        self.transform = transform
        
        # Build a list of (center_frame_path, pulse_id) for each pulse
        self.samples = []
        for pulse_id in pulse_list:
            pulse_dir = os.path.join(self.frames_root, f"pulse{pulse_id}")
            frame_paths = sorted(glob.glob(os.path.join(pulse_dir, "*.png")))
            
            if len(frame_paths) == 0:
                # Skip pulses with no frames
                print(f"Skipping pulse {pulse_id} with no frames")
                continue
            # Get middle frame index
            mid_idx = len(frame_paths) // 2  # floor division
            # e.g. if 30 frames, mid_idx = 15 => 0-based indexing => 16th frame
            # if 31 frames, mid_idx = 15 => 16th frame => the center
            center_frame_path = frame_paths[mid_idx]
            
            self.samples.append((center_frame_path, pulse_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, pulse_id = self.samples[idx]
        
        # Load & transform the image
        image = Image.open(frame_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Extract betas for this pulse
        # masked_data: (9423, Npulses)
        # pulse_id is 1-based, so subtract 1 for 0-based indexing
        betas_1d = self.masked_data[:, pulse_id - 1]  # shape: (9423,)
        
        # Convert to torch
        betas_1d = torch.from_numpy(betas_1d)
        # Expand to (1, 9423) => (batch, seq_len, #voxels)-like shape
        betas_2d = betas_1d.unsqueeze(0)
        
        return {
            "image": image,
            "voxel": betas_2d,   # shape: (1, 9423)
            "pulse_id": pulse_id,
            "frame_path": frame_path
        }


train_dataset = MovieBetasDataset(
    masked_data=masked_movie_data,
    frames_root=frames_data_path,
    pulse_list=train_pulses,
    transform=transform
)

val_dataset = MovieBetasDataset(
    masked_data=masked_movie_data,
    frames_root=frames_data_path,
    pulse_list=val_pulses,
    transform=transform
)


test_dataset = MovieBetasDataset(
    masked_data=masked_movie_data,
    frames_root=frames_data_path,
    pulse_list=test_pulses,
    transform=transform
)



train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=per_device_batch_size, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size, num_workers=num_workers, pin_memory=True)







print(f"Train loader: {len(train_loader)} batches")
print(f"Val loader: {len(val_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")


# #print shape of a single batch
# print("Single batch shape:")
# for batch in test_loader:
#     #print avaliable keys in the batch
#     print(batch.keys())
#     print(batch['image'].shape)
#     print(batch['voxel'].shape)
#     print(batch['pulse_id'].shape)
#     break


#Define default vars for training loop here

use_image_aug = False
clip_scale = 1.0
use_prior = True
blurry_recon = True
mixup_pct = 0.33
prior_scale = 30
blur_scale = 0.5
num_voxels = 9423 # The single input size
hidden_dim = 1024 # The hidden dimension size





global_batch_size = per_device_batch_size * num_devices
num_epochs = 1000
num_samples_per_epoch = len(train_loader) / num_devices
max_lr = 3e-4



clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
    cache_dir = cache_dir
)


clip_img_embedder.to(device)
print("CLIP image embedder loaded")

clip_seq_dim = 256
clip_emb_dim = 1664

autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=256,
)
ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')

autoenc.load_state_dict(ckpt)

autoenc.eval()
autoenc.requires_grad_(False)
autoenc = autoenc.to(device)

print("Autoencoder loaded")
utils.count_params(autoenc)

cnx = ConvnextXL(f'/home/csasi/True_Hallucinations_link/data/Mindeye_NSD_data/cache_dir/convnext_xlarge_alpha0.75_fullckpt.pth')
cnx.requires_grad_(False)
cnx.eval()
cnx = cnx.to(device)


mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)

blur_augs = AugmentationSequential(
    kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
    kornia.augmentation.RandomGrayscale(p=0.1),
    kornia.augmentation.RandomSolarize(p=0.1),
    kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
    data_keys=["input"],
)

# 1) Initialize MindEye2 model
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()

# MindEye2's linear layer: map 9423 voxels -> hidden_dim

# Simplified RidgeRegression (No subj_idx, No reshape fix)
class RidgeRegression(nn.Module):
    def __init__(self, input_size, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linear = nn.Linear(input_size, out_features)
        print(f"Initialized Single-Subject RidgeRegression: input_size={input_size}, out_features={out_features}")

    def forward(self, x):
        B, _, V = x.shape
        x = x.view(B, V) # Reshape input: [B, V]
        # --- Direct linear call ---
        out = self.linear(x)
        out = out.unsqueeze(1) # Add back sequence dimension: [B, 1, out_features]
        return out




model.ridge = RidgeRegression(input_size=num_voxels, out_features=hidden_dim)



utils.count_params(model.ridge)
utils.count_params(model)

b = torch.randn((2,1,num_voxels))
print("b.shape",b.shape)


# MindEye2's backbone
model.backbone = BrainNetwork(
    h=hidden_dim,
    in_dim=hidden_dim,
    seq_len=1,
    n_blocks=n_blocks,
    clip_size=clip_emb_dim,
    out_dim=clip_emb_dim * clip_seq_dim,  # depends on architecture
    blurry_recon=blurry_recon, 
    clip_scale=clip_scale,
)
# For your backbone model


utils.count_params(model.backbone)
utils.count_params(model)

# test that the model works on some fake data
b = torch.randn((2,1,hidden_dim))
print("b.shape",b.shape)

backbone_, clip_, blur_ = model.backbone(b)
print(backbone_.shape, clip_.shape, blur_[0].shape, blur_[1].shape)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']


def save_ckpt(tag):
    ckpt_path = output_dir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {output_dir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,output_dir=output_dir,multisubj_loading=False): 
    print(f"\n---loading {output_dir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(output_dir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

print("\nDone with model preparations!")
num_params = utils.count_params(model)

seed = 999
utils.seed_everything(seed)


model_name = "MindEye2_movie"

wandb_project = 'MindEye2_movie'
print(f"wandb {wandb_project} run {model_name}")
# need to configure wandb beforehand in terminal with "wandb init"!
wandb_config = {
    "model_name": model_name,
    "global_batch_size": global_batch_size,
    "batch_size": global_batch_size,
    "num_epochs": num_epochs,
    "num_params": num_params,
    "clip_scale": clip_scale,
    "prior_scale": prior_scale,
    "blur_scale": blur_scale,
    "use_image_aug": use_image_aug,
    "max_lr": max_lr,
    "mixup_pct": mixup_pct,
    "num_samples_per_epoch": num_samples_per_epoch,
    "ckpt_interval": 100,
    "seed": seed,
    "num_devices": num_devices,
    "world_size": world_size,

}
print("wandb_config:\n",wandb_config)
print("wandb_id:",model_name)
wandb.init(
    id=model_name,
    project=wandb_project,
    name=model_name,
    config=wandb_config,
    resume="allow",
)






opt_grouped_parameters = [
    {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]




torch.cuda.empty_cache()
epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9



model_name = "MindEye2_movie"
print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

num_iterations_per_epoch = len(train_loader)
original_print = print

def rank0_print(*args, **kwargs):
    if rank == 0:
        original_print(*args, **kwargs)

print = rank0_print





optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

lr_scheduler_type = 'cycle'

total_steps= num_epochs * num_iterations_per_epoch
print("total_steps", total_steps)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=max_lr,
    total_steps=total_steps,
    final_div_factor=1000,
    last_epoch=-1, pct_start=2/num_epochs
)





# Instead of FSDP, wrap via accelerate:
model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader, test_loader)



print = accelerator.print


progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(rank != 0))

for epoch in progress_bar:


        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.
        
        recon_cossim = 0.
        test_recon_cossim = 0.
        recon_mse = 0.
        test_recon_mse = 0.

        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.
        
        loss_prior_total = 0.
        test_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

        # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)


        # Pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
        num_iterations_per_epoch = len(train_loader)
        voxel_iters = {}  # Will store voxel data by iteration
        image_iters = torch.zeros(num_iterations_per_epoch, per_device_batch_size, 3, 224, 224).float()
    
        # Optional dictionaries for mixup if needed
        perm_iters, betas_iters, select_iters = {}, {}, {}

        for iter, batch in enumerate(train_loader):
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=data_type):
        
                # Get image and voxel directly from the batch dictionary
                image0 = batch['image']  # Shape [1, 3, 224, 224]
                voxel0 = batch['voxel']  # Shape [1, 1, 9423]
                
                assert image0.shape[0] == per_device_batch_size, f"Batch size mismatch: got {image0.shape[0]}, expected {per_device_batch_size}"
                
                # Store the image
                image_iters[iter] = image0
                
                # Apply mixup if needed
                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
               
                    # Store mixup parameters
                    perm_iters[f"iter{iter}"] = perm
                    betas_iters[f"iter{iter}"] = betas
                    select_iters[f"iter{iter}"] = select

                # Store the voxel data
                voxel_iters[f"iter{iter}"] = voxel0

                if iter >= num_iterations_per_epoch-1:
                    break

        for train_i in range(num_iterations_per_epoch):
            #with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=data_type):    
            with torch.amp.autocast(device_type='cuda', dtype=data_type):
        
                optimizer.zero_grad()
                loss = 0.
                voxel_list = voxel_iters[f"iter{train_i}"].detach().to(device)
                image = image_iters[train_i].detach()
                image = image.to(device)
                print(f"Shape and sample of image")
                print(image.shape)

                clip_target = clip_img_embedder(image)
                
                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(mixup_pct * num_epochs):
                    perm = perm_iters[f"iter{train_i}"].detach().to(device)
                    betas = betas_iters[f"iter{train_i}"].detach().to(device)
                    select = select_iters[f"iter{train_i}"].detach().to(device)

                voxel_list = voxel_list.to(device)


                print("voxel_list.shape", voxel_list.shape)

                voxel_ridge = model.ridge(voxel_list)


                # Process through backbone
                backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

                # print(backbone_.shape, clip_voxels.shape, blurry_image_enc_.shape)
                # print("backbone_.device", backbone_.device, "clip_voxels.device", clip_voxels.device, "blurry_image_enc_.device", blurry_image_enc_.device)
                
                # Normalize CLIP representations if using CLIP loss
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                
                loss_prior_total += loss_prior.item()
                loss_prior *= prior_scale
                loss += loss_prior

                recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                recon_mse += mse(prior_out, clip_target).item()

                if epoch < int(mixup_pct * num_epochs):                
                            loss_clip = utils.mixco_nce(
                                clip_voxels_norm,
                                clip_target_norm,
                                temp=.006,
                                perm=perm, betas=betas, select=select)
            
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)
            
                loss_clip_total += loss_clip.item()
                loss_clip *= clip_scale
                loss += loss_clip



                image_enc_pred, transformer_feats = blurry_image_enc_
                # print(f"image_enc_pred data type", image_enc_pred.dtype)
                # print(f"image_enc_pred device", image_enc_pred.device)
                # print(f"image_enc_pred shape", image_enc_pred.shape)
                # print(f"transformer_feats data type", type(transformer_feats))
                # print(f"transformer_feats device", transformer_feats.device)
                # print(f"transformer_feats shape", transformer_feats.shape)
               
                # move image_enc_pred to the same device as image
#                image_enc_pred = image_enc_pred.to(device)
                # print("image_enc_pred.shape:", image_enc_pred.shape, "device:", image_enc_pred.device)
                # print("transformer_feats.shape", transformer_feats.shape)
                # print("transformer_feats.device", transformer_feats.device)
    
                image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                image_enc = image_enc.to(device)                        
                loss_blurry = l1(image_enc_pred, image_enc)
                loss_blurry_total += loss_blurry.item()


                if epoch < int(mixup_pct * num_epochs):
                    image_enc_shuf = image_enc[perm]
                    betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                    
                    # print("betas_shape", betas_shape)
                    # print("betas data type", betas.dtype)
                    # print("betas device", betas.device)

                    #convert betas to bf16
                    betas = betas.to(data_type)
                    # print("betas data type", betas.dtype)

                    image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                        image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                image_norm = (image - mean)/std
                image_aug = (blur_augs(image) - mean)/std
                
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)
                # print("cnx_embeds.shape", cnx_embeds.shape)
                # print("cnx_embeds.device", cnx_embeds.device)
                # print("cnx_aug_embeds.shape", cnx_aug_embeds.shape)
                # print("cnx_aug_embeds.device", cnx_aug_embeds.device)
                
                # cnx_embeds = cnx_embeds.to(device)
                # cnx_aug_embeds = cnx_aug_embeds.to(device)

                cont_loss = utils.soft_cont_loss(
                    nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.2)
                
                loss_blurry_cont_total += cont_loss.item()
                loss += (loss_blurry + 0.1*cont_loss) * blur_scale #/.18215

                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()


        


                with torch.no_grad():
                    # Select samples, ensuring at least one sample is selected
                    random_samps = np.random.choice(np.arange(len(image)), size=max(1, len(image)//2), replace=False)
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    blurry_pixcorr += pixcorr.item()

                    
                utils.check_loss(loss)
                # Standard backward pass with monitoring
                with torch.amp.autocast('cuda', dtype=data_type):
                    accelerator.backward(loss)

                print(f"Epoch {epoch} iter {train_i+1}/{num_iterations_per_epoch} "
                            f"Train loss: {loss.item():.4f}, ")


   

                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])
                
                if lr_scheduler_type is not None:
                    lr_scheduler.step()


                

        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=data_type):

            # Collect all test samples
            all_images = []
            all_voxels = []

            # Collect all test samples
            for batch in val_loader:
                image = batch['image'].to(device)  # Shape [1, 3, 224, 224]
                voxel = batch['voxel'].to(device)  # Shape [1, 1, 9423]
                all_images.append(image)
                all_voxels.append(voxel)

            # Stack all collected samples
            all_images = torch.cat(all_images, dim=0)
            all_voxels = torch.cat(all_voxels, dim=0)

            # Limit to first 200 samples for evaluation
            test_indices = torch.arange(min(len(all_voxels), 200), device=device)
            voxel = all_voxels[test_indices]
            image = all_images[test_indices]
            
            # Create 3 repetitions of voxel data for averaging
            # Since we don't have multiple representations, we'll duplicate each voxel
            voxel_expanded = voxel.repeat(1, 3, 1)  # Create 3 copies in dimension 1
            
            clip_target = clip_img_embedder(image)
            
            # Process each representation and average the results
            clip_voxels = None
            backbone = None
            
            #Print all details of voxel_expanded[:, rep]
            print("voxel_expanded[:, rep]", voxel_expanded[:, 0].shape)
            print(" voxel _expanded device", voxel_expanded[:, 0].device)
            print("voxel_expanded[:, rep] dtype", voxel_expanded[:, 0].dtype)
            
            for rep in range(3):
                # Extract the rep-th representation from each sample
                voxel_rep = voxel_expanded[:, rep].unsqueeze(1)  # Shape [batch_size, 1, 9423]
                voxel_ridge = model.ridge(voxel_rep)  # 0th index of subj_list
                backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
                
                if rep == 0:
                    clip_voxels = clip_voxels0
                    backbone = backbone0
                else:
                    clip_voxels += clip_voxels0
                    backbone += backbone0
                    
            clip_voxels /= 3
            backbone /= 3
        

            # Normalize CLIP representations
            clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
            
            # For some evals, select a subset of samples
            num_random_samples = max(1, len(image) // 5)

            random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)

            
            chunk_size = 20

            # Accumulators
            total_prior_loss = 0.0
            total_cossim = 0.0
            total_mse_val = 0.0
            total_samples = 0

            for i in range(0, len(random_samps), chunk_size):
                subset_indices = random_samps[i : i + chunk_size]

                loss_prior_chunk, cont_out_chunk = model.diffusion_prior(
                    text_embed=backbone[subset_indices],
                    image_embed=clip_target[subset_indices]
                )

                # Accumulate scaled diffusion prior loss
                total_prior_loss += loss_prior_chunk.item() * prior_scale

                # Compute partial metrics on just this chunk
                # (no need to concatenate & store all chunk outputs)
                target_chunk = clip_target[subset_indices]
                total_cossim += nn.functional.cosine_similarity(
                    cont_out_chunk, target_chunk
                ).sum().item()  # sum, not mean

                total_mse_val += nn.functional.mse_loss(
                    cont_out_chunk, target_chunk, reduction='sum'
                ).item()  # sum, not mean

                total_samples += cont_out_chunk.shape[0]

                # Optionally free memory from chunk outputs
                del cont_out_chunk, target_chunk, loss_prior_chunk

            # Now compute average metrics
            test_loss_prior_total += total_prior_loss
            mean_cossim = total_cossim / total_samples
            mean_mse_val = total_mse_val / total_samples

            # If desired, add these aggregated metrics to your logs
            test_recon_cossim += mean_cossim
            test_recon_mse += mean_mse_val
            print(f"Mean Cosine Similarity: {mean_cossim:.4f}, Mean MSE: {mean_mse_val:.4f}")


            #get loss prior from model.diffusion_prior
            loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
            test_loss_prior_total += loss_prior.item()
            loss_prior *= prior_scale
            loss += loss_prior

            #Get clip loss        
            loss_clip = utils.soft_clip_loss(
                clip_voxels_norm,
                clip_target_norm,
                temp=.006)

            test_loss_clip_total += loss_clip.item()
            loss_clip = loss_clip * clip_scale
            loss += loss_clip


            # Initialize accumulators
            loss_prior_total_local = 0.
            cont_out_list = []

            for i in range(0, len(random_samps), chunk_size):
                subset_indices = random_samps[i : i + chunk_size]

                # text_embed and image_embed are partial
                loss_prior_chunk, cont_out_chunk = model.diffusion_prior(
                    text_embed=backbone[subset_indices],
                    image_embed=clip_target[subset_indices]
                )
                # Accumulate the loss
                loss_prior_total_local += loss_prior_chunk.item()

                # Save the predictions if you need them for metrics
                cont_out_list.append(cont_out_chunk)    

            # Convert list of chunked outputs back to a single tensor for metrics
            contaminated_prior_out = torch.cat(cont_out_list, dim=0)  # shape = [N, embed_dim]

            # If you want to apply prior_scale after summing all chunked losses:
            loss_prior = loss_prior_total_local * prior_scale
            loss += loss_prior
            
            # Calculate metrics
            test_recon_cossim += nn.functional.cosine_similarity(contaminated_prior_out, clip_target[random_samps]).mean().item()
            test_recon_mse += mse(contaminated_prior_out, clip_target[random_samps]).item()
            
            loss_clip = utils.soft_clip_loss(clip_voxels_norm,clip_target_norm,temp=0.006)
                
            test_loss_clip_total += loss_clip.item()
            loss_clip = loss_clip * clip_scale
            loss += loss_clip
            
            # Calculate accuracy metrics
            labels = torch.arange(len(clip_voxels_norm), device=device)
            test_fwd_percent_correct += utils.topk(
                utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
            test_bwd_percent_correct += utils.topk(
                utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), 
                labels, 
                k=1
            ).item()

            image_enc_pred, _ = blurry_image_enc_
            blurry_recon_images = (autoenc.decode(
                image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5
            ).clamp(0, 1)
            pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
            test_blurry_pixcorr += pixcorr.item()
        
            utils.check_loss(loss)
            test_losses.append(loss.item())

            test_i = len(test_losses)
            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                "train/recon_cossim": recon_cossim / (train_i + 1),
                "test/recon_cossim": test_recon_cossim / (test_i + 1),
                "train/recon_mse": recon_mse / (train_i + 1),
                "test/recon_mse": test_recon_mse / (test_i + 1),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / (test_i + 1),
                }
            







# save the plot os loss and test loss
time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, f"{model_name}_{time}_epoch{epoch}.png"))
