#!/bin/bash

### parameters
dataset=OfficeHome
gpu=0
batch_size=20
ddim_steps=50
iter_per_class=800 
int_lower=0.3
int_upper=0.7
cfg_lower=4.0
cfg_upper=6.0
class_name=("Alarm_Clock" "Backpack" "Batteries" "Bed" "Bike" "Bottle" "Bucket" "Calculator" "Calendar" "Candles" "Chair" "Clipboards" "Computer" "Couch" "Curtains" "Desk_Lamp" "Drill" "Eraser" "Exit_Sign" "Fan" "File_Cabinet" "Flipflops" "Flowers" "Folder" "Fork" "Glasses" "Hammer" "Helmet" "Kettle" "Keyboard" "Knives" "Lamp_Shade" "Laptop" "Marker" "Monitor" "Mop" "Mouse" "Mug" "Notebook" "Oven" "Pan" "Paper_Clip" "Pen" "Pencil" "Postit_Notes" "Printer" "Push_Pin" "Radio" "Refrigerator" "Ruler" "Scissors" "Screwdriver" "Shelf" "Sink" "Sneakers" "Soda" "Speaker" "Spoon" "TV" "Table" "Telephone" "ToothBrush" "Toys" "Trash_Can" "Webcam")


########################## interploation using the DM trained on "Art Clipart Product" domains => indexes: "0" "1" "2"
source_domains="012"
training_domains=("Art" "Clipart" "Product") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "Art" and "Clipart"
int_domains=("Art" "Clipart")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Clipart" and "Product"
int_domains=("Clipart" "Product")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Art" and "Product"
int_domains=("Art" "Product")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"






########################## interploation using the DM trained on "Art Clipart Real World" domains  => indexes: "0" "1" "3"
source_domains="013"
training_domains=("Art" "Clipart" "Real World") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "Art" and "Clipart"
int_domains=("Art" "Clipart")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Clipart" and "Real World"
int_domains=("Clipart" "Real World")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Art" and "Real World"
int_domains=("Art" "Real World")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"






########################## interploation using the DM trained on "Art Product Real World" domains => indexes: "0" "2" "3"
source_domains="023"
training_domains=("Art" "Product" "Real World") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "Art" and "Product"
int_domains=("Art" "Product")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Product" and "Real World"
int_domains=("Product" "Real World")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Art" and "Real World"
int_domains=("Art" "Real World")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"






########################## interploation using the DM trained on "Clipart Product Real World" domains => indexes: "1" "2" "3"
source_domains="123"
training_domains=("Clipart" "Product" "Real World") # corresponding names for the source_domains
config_dir="configs/${dataset}/d${source_domains}.yaml"
save_dir="save/dm/${dataset}/${source_domains}/generation"
ckpt_dir="path/to/a/checkpoint" # path to the trained model



#interpolating "Clipart" and "Product"
int_domains=("Clipart" "Product")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Product" and "Real World"
int_domains=("Product" "Real World")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"


#interpolating "Clipart" and "Real World"
int_domains=("Clipart" "Real World")
CUDA_VISIBLE_DEVICES=${gpu}  python interpolation.py --training_domains "${training_domains[@]}" --int_domains "${int_domains[@]}" --int_bounds ${int_lower} ${int_upper} --scale_bounds ${cfg_lower} ${cfg_upper} --outdir ${save_dir} --H 256 --W 256 --n_samples ${batch_size} --iter_per_class ${iter_per_class} --config ${config_dir} --ckpt ${ckpt_dir} --ddim_steps ${ddim_steps}  --classes "${class_name[@]}"




