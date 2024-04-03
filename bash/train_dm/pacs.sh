#!/bin/bash

export NCCL_BLOCKING_WAIT=1
nvidia-smi


### parameters
dataset=PACS
scale_lr=False
no_test=True
num_nodes=1
check_val_every_n_epoch=5
gpus="0,1,2,3"
init_weights="init_weights/v1-5-pruned.ckpt"



### training for PACS dataset, 012 domain indexes as source domains (art, cartoon, photo)
source_domains="012"
config_dir="configs/${dataset}/d${source_domains}.yaml"
logdir="save/dm/${dataset}/${source_domains}/"

python train_dm.py -t --base ${config_dir} --gpus ${gpus} --scale_lr ${scale_lr} --no-test ${no_test} --num_nodes ${num_nodes} --check_val_every_n_epoch ${check_val_every_n_epoch} --logdir ${logdir} --init_weights ${init_weights}




### training for PACS dataset, 013 domain indexes as source domains (art, cartoon, sketch)
source_domains="013"
config_dir="configs/${dataset}/d${source_domains}.yaml"
logdir="save/dm/${dataset}/${source_domains}/"

python train_dm.py -t --base ${config_dir} --gpus ${gpus} --scale_lr ${scale_lr} --no-test ${no_test} --num_nodes ${num_nodes} --check_val_every_n_epoch ${check_val_every_n_epoch} --logdir ${logdir} --init_weights ${init_weights}




### training for PACS dataset, 023 domain indexes as source domains (art, photo, sketch)
source_domains="023"
config_dir="configs/${dataset}/d${source_domains}.yaml"
logdir="save/dm/${dataset}/${source_domains}/"

python train_dm.py -t --base ${config_dir} --gpus ${gpus} --scale_lr ${scale_lr} --no-test ${no_test} --num_nodes ${num_nodes} --check_val_every_n_epoch ${check_val_every_n_epoch} --logdir ${logdir} --init_weights ${init_weights}




### training for PACS dataset, 123 domain indexes as source domains (cartoon, photo, sketch)
source_domains="123"
config_dir="configs/${dataset}/d${source_domains}.yaml"
logdir="save/dm/${dataset}/${source_domains}/"

python train_dm.py -t --base ${config_dir} --gpus ${gpus} --scale_lr ${scale_lr} --no-test ${no_test} --num_nodes ${num_nodes} --check_val_every_n_epoch ${check_val_every_n_epoch} --logdir ${logdir} --init_weights ${init_weights}

