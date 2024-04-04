#!/bin/bash


### parameters
dataset=PACS
save_dir="./save/train_cls_fds/${dataset}/"
data_dir="./data"
gen_num_per_class=570 # number of images/class of the interpolated (generated) data that will be selecte to used in training



########################## training the classifier using both original and interpolated data using the source domains: "art_painting cartoon photo" domains => indexes: "0" "1" "2"
source_domains="012"
test_env=3
gen_data_dir="save/dm/${dataset}/${source_domains}/generation"

### first seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed0/image_predictions.csv"
python train_cls.py ${dataset}0_${source_domains} --dataset ${dataset} --deterministic --trial_seed 0 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### second seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed1/image_predictions.csv"
python train_cls.py ${dataset}1_${source_domains} --dataset ${dataset} --deterministic --trial_seed 1 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### third seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed2/image_predictions.csv"
python train_cls.py ${dataset}2_${source_domains} --dataset ${dataset} --deterministic --trial_seed 2 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct



########################## training the classifier using both original and interpolated data using the source domains: "art_painting cartoon sketch" domains => indexes: "0" "1" "3"
source_domains="013"
test_env=2
gen_data_dir="save/dm/${dataset}/${source_domains}/generation"

### first seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed0/image_predictions.csv"
python train_cls.py ${dataset}0_${source_domains} --dataset ${dataset} --deterministic --trial_seed 0 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### second seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed1/image_predictions.csv"
python train_cls.py ${dataset}1_${source_domains} --dataset ${dataset} --deterministic --trial_seed 1 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### third seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed2/image_predictions.csv"
python train_cls.py ${dataset}2_${source_domains} --dataset ${dataset} --deterministic --trial_seed 2 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct



########################## training the classifier using both original and interpolated data using the source domains: "art_painting photo sketch" domains => indexes: "0" "2" "3"
source_domains="023" 
test_env=1
gen_data_dir="save/dm/${dataset}/${source_domains}/generation"

### first seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed0/image_predictions.csv"
python train_cls.py ${dataset}0_${source_domains} --dataset ${dataset} --deterministic --trial_seed 0 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### second seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed1/image_predictions.csv"
python train_cls.py ${dataset}1_${source_domains} --dataset ${dataset} --deterministic --trial_seed 1 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### third seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed2/image_predictions.csv"
python train_cls.py ${dataset}2_${source_domains} --dataset ${dataset} --deterministic --trial_seed 2 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct



########################## training the classifier using both original and interpolated data using the source domains: "cartoon photo sketch" domains => indexes: "1" "2" "3"
source_domains="123"
test_env=0
gen_data_dir="save/dm/${dataset}/${source_domains}/generation"

### first seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed0/image_predictions.csv"
python train_cls.py ${dataset}0_${source_domains} --dataset ${dataset} --deterministic --trial_seed 0 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### second seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed1/image_predictions.csv"
python train_cls.py ${dataset}1_${source_domains} --dataset ${dataset} --deterministic --trial_seed 1 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct


### third seed
gen_csv_dir="save/eval/${dataset}/${source_domains}/seed2/image_predictions.csv"
python train_cls.py ${dataset}2_${source_domains} --dataset ${dataset} --deterministic --trial_seed 2 --checkpoint_freq 100 --test_envs ${test_env} --data_dir ${data_dir} --work_dir $save_dir --use_gen --gen_data_dir ${gen_data_dir} --gen_csv_dir ${gen_csv_dir} --gen_num_per_class ${gen_num_per_class} --gen_only_correct

