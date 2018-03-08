#!/usr/bin/env bash
python run.py --gpu 2 --num_epoch_for_offline 100 --num_epoch_for_online 10 --lr_stop 1e-7 --fixed_conv 0 --momentum 0.9