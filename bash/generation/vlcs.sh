#!/bin/bash

### parameters
dataset=VLCS
batch_size=20
ddim_steps=50
iter_per_class=1600 
int_lower=0.3
int_upper=0.7
cfg_lower=4.0
cfg_upper=6.0
class_name=("bird" "car" "chair" "dog" "person") # classes of VLCS


########################## interploation using the DM trained on "Caltech101 LabelMe SUN09" domains
source_domains="012"
training_domains=("Caltech101" "LabelMe" "SUN09") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "Caltech101" and "LabelMe"
augment_domains=("Caltech101" "LabelMe")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "LabelMe" and "SUN09"
augment_domains=("LabelMe" "SUN09")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "Caltech101" and "SUN09"
augment_domains=("Caltech101" "SUN09")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}






########################## interploation using the DM trained on "Caltech101 LabelMe VOC2007" domains
source_domains="013"
training_domains=("Caltech101" "LabelMe" "VOC2007") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "Caltech101" and "LabelMe"
augment_domains=("Caltech101" "LabelMe")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "LabelMe" and "VOC2007"
augment_domains=("LabelMe" "VOC2007")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "Caltech101" and "VOC2007"
augment_domains=("Caltech101" "VOC2007")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}






########################## interploation using the DM trained on "Caltech101 SUN09 VOC2007" domains
source_domains="023"
training_domains=("Caltech101" "SUN09" "VOC2007") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "Caltech101" and "SUN09"
augment_domains=("Caltech101" "SUN09")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "SUN09" and "VOC2007"
augment_domains=("SUN09" "VOC2007")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "Caltech101" and "VOC2007"
augment_domains=("Caltech101" "VOC2007")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}






########################## interploation using the DM trained on "LabelMe SUN09 VOC2007" domains
source_domains="123"
training_domains=("LabelMe" "SUN09" "VOC2007") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "LabelMe" and "SUN09"
augment_domains=("LabelMe" "SUN09")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "SUN09" and "VOC2007"
augment_domains=("SUN09" "VOC2007")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}


#interpolating "LabelMe" and "VOC2007"
augment_domains=("LabelMe" "VOC2007")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes ${class_name}




