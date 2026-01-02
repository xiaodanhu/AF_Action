#!/bin/bash
source /data/xiaodan8/anaconda3/bin/activate parsing2

deepspeed --include="localhost:1,3" --master_port 12366 train_uniform_thumos14.py --round 500 &&
deepspeed --include="localhost:1,3" --master_port 12366 eval_shard.py --config ./configs/thumos_i3d_uniform.yaml --ckpt ./ckpt/thumos_i3d_uniform_ds --filename vit_uniform_500
#  &&
# deepspeed --include="localhost:0,1" --master_port 12366 train_uniform_thumos14.py --resume vit_uniform_500 --round 1000 &&
# deepspeed --include="localhost:0,1" --master_port 12366 eval_shard.py --config ./configs/thumos_i3d_uniform.yaml --ckpt ./ckpt/thumos_i3d_uniform_ds --filename vit_uniform_1000 &&
# deepspeed --include="localhost:0,1" --master_port 12366 train_uniform_thumos14.py --resume vit_uniform_1000 --round 1500 &&
# deepspeed --include="localhost:0,1" --master_port 12366 eval_shard.py --config ./configs/thumos_i3d_uniform.yaml --ckpt ./ckpt/thumos_i3d_uniform_ds --filename vit_uniform_1500 &&
# deepspeed --include="localhost:0,1" --master_port 12366 train_uniform_thumos14.py --resume vit_uniform_1500 --round 2000 &&
# deepspeed --include="localhost:0,1" --master_port 12366 eval_shard.py --config ./configs/thumos_i3d_uniform.yaml --ckpt ./ckpt/thumos_i3d_uniform_ds --filename vit_uniform_2000 &&
# deepspeed --include="localhost:0,1" --master_port 12366 train_uniform_thumos14.py --resume vit_uniform_2000 --round 2500 &&
# deepspeed --include="localhost:0,1" --master_port 12366 eval_shard.py --config ./configs/thumos_i3d_uniform.yaml --ckpt ./ckpt/thumos_i3d_uniform_ds --filename vit_uniform_2500



deepspeed --include="localhost:2,3" --master_port 12109 train_uniform_thumos14.py --resume vit_uniform_500 --round 800 &&
deepspeed --include="localhost:2,3" --master_port 12109 train_uniform_thumos14.py --resume vit_uniform_800 --round 1100 &&
deepspeed --include="localhost:2,3" --master_port 12109 train_uniform_thumos14.py --resume vit_uniform_1100 --round 1400 &&
deepspeed --include="localhost:2,3" --master_port 12109 train_uniform_thumos14.py --resume vit_uniform_1400 --round 1700 &&
deepspeed --include="localhost:2,3" --master_port 12109 train_uniform_thumos14.py --resume vit_uniform_1700 --round 2000 &&
deepspeed --include="localhost:2,3" --master_port 12109 train_uniform_thumos14.py --resume vit_uniform_2000 --round 2300