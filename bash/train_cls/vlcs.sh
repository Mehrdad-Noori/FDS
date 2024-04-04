#!/bin/bash


### parameters
dataset=VLCS
save_dir="./save/train_cls/${dataset}/"
data_dir="./data"



### first seed
python train_cls.py ${dataset}0 --dataset ${dataset} --deterministic --trial_seed 0 --checkpoint_freq 50 --tolerance_ratio 0.2 ${data_dir} --work_dir $save_dir

### second seed
python train_cls.py ${dataset}1 --dataset ${dataset} --deterministic --trial_seed 1 --checkpoint_freq 50 --tolerance_ratio 0.2 ${data_dir} --work_dir $save_dir

### third seed
python train_cls.py ${dataset}2 --dataset ${dataset} --deterministic --trial_seed 2 --checkpoint_freq 50 --tolerance_ratio 0.2 ${data_dir} --work_dir $save_dir
