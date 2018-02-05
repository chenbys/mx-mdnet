#!/usr/bin/env bash
python train.py --gpu 1 --loss_type 1 --fixed_conv 0 --batch_callback_freq 50 --momentum 0.5 --num_epoch 20 --lr 5e-5 --wd 1e-0