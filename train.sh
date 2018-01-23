#!/usr/bin/env bash
python train.py --gpu 3 --lr 1e-7 --wd 5e-4 --num_epoch 100 --batch_callback_freq 50 --lr_step 72 --lr_stop 1e-12 --momentum 0.9