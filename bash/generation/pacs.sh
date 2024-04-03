#!/bin/bash

### parameters
dataset=PACS
batch_size=20
ddim_steps=50
iter_per_class=1600 
int_lower=0.3
int_upper=0.7
cfg_lower=4.0
cfg_upper=6.0
class_name=("dog" "guitar" "horse" "elephant" "house" "person" "giraffe") # classes of PACS


########################## interploation using the DM trained on "art_painting cartoon photo" domains
source_domains="012"
training_domains=("art_painting" "cartoon" "photo") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "art_painting" and "cartoon"
augment_domains=("art_painting" "cartoon")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "cartoon" and "photo"
augment_domains=("cartoon" "photo")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "art_painting" and "photo"
augment_domains=("art_painting" "photo")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"






########################## interploation using the DM trained on "art_painting cartoon sketch" domains
source_domains="013"
training_domains=("art_painting" "cartoon" "sketch") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "art_painting" and "cartoon"
augment_domains=("art_painting" "cartoon")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "cartoon" and "sketch"
augment_domains=("cartoon" "sketch")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "art_painting" and "sketch"
augment_domains=("art_painting" "sketch")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"






########################## interploation using the DM trained on "art_painting photo sketch" domains
source_domains="023"
training_domains=("art_painting" "photo" "sketch") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "art_painting" and "photo"
augment_domains=("art_painting" "photo")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "photo" and "sketch"
augment_domains=("photo" "sketch")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "art_painting" and "sketch"
augment_domains=("art_painting" "sketch")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"






########################## interploation using the DM trained on "cartoon photo sketch" domains
source_domains="123"
training_domains=("cartoon" "photo" "sketch") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "cartoon" and "photo"
augment_domains=("cartoon" "photo")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "photo" and "sketch"
augment_domains=("photo" "sketch")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "cartoon" and "sketch"
augment_domains=("cartoon" "sketch")
python interpolation.py --training_domains "${training_domains[@]}" --augment_domains "${augment_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"




