#!/bin/bash


### parameters
dataset=PACS
save_dir="./save/dm/${dataset}/"
data_dir="./data"



### first seed
python train_cls.py ${dataset}0 --dataset ${dataset} --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ${data_dir} --work_dir $save_dir

### second seed
python train_cls.py ${dataset}1 --dataset ${dataset} --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir ${data_dir} --work_dir $save_dir

### third seed
python train_cls.py ${dataset}2 --dataset ${dataset} --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir ${data_dir} --work_dir $save_dir
