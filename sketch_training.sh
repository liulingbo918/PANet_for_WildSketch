#!/bin/bash

name=WildSketch_Experiment

python train.py \
			--dataset WildSketch \
			--name $name \
			--model sketch \
			--direction AtoB \
			--dataset_mode sketch \
			--output_nc 1 \
			--load_size 256 \
			--crop_size 256 \
			--preprocess resize_and_crop \
			--input_nc 3 \
			--output_nc 1 \
			--display_env $name \
			--batch_size 1 \
			--print_freq 10 \
			--norm instance \
			--netG PANet \
			--gan 1 \
			--adcnn 1 \
			--ngf 64 \
			--splits 3 \
			--splits 4 \
			--splits 5 \
			--FMN 256 \
			--FMN 512 \
			--group_fc 32 \
			--adc 32 \
			--dfcnn 1 \
			--lambda_L1 100 \
			--lambda_SSIM 100 \
			--shortcut_adaptive 1 \
			--niter 200 \
			--niter_decay 200 \
			--lr_policy 'step'\
			--lr_decay_iters 200 \
			--gpu_ids 0 \
			--save_epoch_freq 1 \
			--display_id 0 \
