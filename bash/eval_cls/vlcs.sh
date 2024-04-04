#!/bin/bash


### 012 domain indexes to get the generated data related to source domains (Caltech101, LabelMe, SUN09)
dataset="VLCS"
source_domains="012"
generated_data_dir="save/dm/${dataset}/${source_domains}/generation"

#seed0
save_dir="save/eval/${dataset}/${source_domains}/seed0"
ckpt_dir="path/to/the/pretrained/model/seed0/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed1
save_dir="save/eval/${dataset}/${source_domains}/seed1"
ckpt_dir="path/to/the/pretrained/model/seed1/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed2
save_dir="save/eval/${dataset}/${source_domains}/seed2"
ckpt_dir="path/to/the/pretrained/model/seed2/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}






### 013 domain indexes to get the generated data related to source domains (Caltech101, LabelMe, VOC2007)
dataset="PACS"
source_domains="013"
generated_data_dir="save/dm/${dataset}/${source_domains}/generation"

#seed0
save_dir="save/eval/${dataset}/${source_domains}/seed0"
ckpt_dir="path/to/the/pretrained/model/seed0/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed1
save_dir="save/eval/${dataset}/${source_domains}/seed1"
ckpt_dir="path/to/the/pretrained/model/seed1/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed2
save_dir="save/eval/${dataset}/${source_domains}/seed2"
ckpt_dir="path/to/the/pretrained/model/seed2/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}







### 023 domain indexes to get the generated data related to source domains (Caltech101, SUN09, VOC2007)
dataset="PACS"
source_domains="023"
generated_data_dir="save/dm/${dataset}/${source_domains}/generation"

#seed0
save_dir="save/eval/${dataset}/${source_domains}/seed0"
ckpt_dir="path/to/the/pretrained/model/seed0/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed1
save_dir="save/eval/${dataset}/${source_domains}/seed1"
ckpt_dir="path/to/the/pretrained/model/seed1/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed2
save_dir="save/eval/${dataset}/${source_domains}/seed2"
ckpt_dir="path/to/the/pretrained/model/seed2/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}







### 123 domain indexes to get the generated data related to source domains (LabelMe, SUN09, VOC2007)
dataset="PACS"
source_domains="123"
generated_data_dir="save/dm/${dataset}/${source_domains}/generation"

#seed0
save_dir="save/eval/${dataset}/${source_domains}/seed0"
ckpt_dir="path/to/the/pretrained/model/seed0/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed1
save_dir="save/eval/${dataset}/${source_domains}/seed1"
ckpt_dir="path/to/the/pretrained/model/seed1/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}

#seed2
save_dir="save/eval/${dataset}/${source_domains}/seed2"
ckpt_dir="path/to/the/pretrained/model/seed2/model.pth"
python eval_cls.py --data_dir $generated_data_dir --save_dir $save_dir --ckpt_dir ${ckpt_dir}