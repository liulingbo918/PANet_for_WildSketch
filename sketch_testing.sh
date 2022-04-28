#!/bin/bash

name=WildSketch_Experiment

python test.py \
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
			--batch_size 1 \
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
			--shortcut_adaptive 1 \
			--gpu_ids 0 \
			--epoch 369
