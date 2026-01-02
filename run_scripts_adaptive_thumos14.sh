#!/bin/bash
source /data/xiaodan8/anaconda3/bin/activate parsing2

# cp -R /data3/xiaodan8/actionformer4_1/ckpt/thumos_i3d_uniform_ds/vit_uniform_500 /data3/xiaodan8/actionformer4_1/ckpt/thumos_i3d_adaptive_ds/vit_adaptive_500 &&
# deepspeed --include="localhost:1,3" --master_port 12108 eval_shard.py --config ./configs/thumos_i3d_adaptive.yaml --ckpt ./ckpt/thumos_i3d_adaptive_ds --filename vit_adaptive_500 &&
# deepspeed --include="localhost:0,1" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_500 --round 1000 &&
# deepspeed --include="localhost:0,1" --master_port 12108 eval_shard.py --config ./configs/thumos_i3d_adaptive.yaml --ckpt ./ckpt/thumos_i3d_adaptive_ds --filename vit_adaptive_1000 &&
# deepspeed --include="localhost:0,1" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_1000 --round 1500 &&
# deepspeed --include="localhost:0,1" --master_port 12108 eval_shard.py --config ./configs/thumos_i3d_adaptive.yaml --ckpt ./ckpt/thumos_i3d_adaptive_ds --filename vit_adaptive_1500 &&
# deepspeed --include="localhost:0,1" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_1500 --round 2000 &&
# deepspeed --include="localhost:0,1" --master_port 12108 eval_shard.py --config ./configs/thumos_i3d_adaptive.yaml --ckpt ./ckpt/thumos_i3d_adaptive_ds --filename vit_adaptive_2000 &&
# deepspeed --include="localhost:0,1" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_2000 --round 2500 &&
# deepspeed --include="localhost:0,1" --master_port 12108 eval_shard.py --config ./configs/thumos_i3d_adaptive.yaml --ckpt ./ckpt/thumos_i3d_adaptive_ds --filename vit_adaptive_2500

# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_500 --round 1000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_1000 --round 1500 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_1500 --round 2000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive_thumos14.py --resume vit_adaptive_2000 --round 2500 &&


deepspeed --include="localhost:1,3" --master_port 12109 train_adaptive_thumos14.py --resume vit_adaptive_500 --round 800 &&
deepspeed --include="localhost:1,3" --master_port 12109 train_adaptive_thumos14.py --resume vit_adaptive_800 --round 1100 &&
deepspeed --include="localhost:1,3" --master_port 12109 train_adaptive_thumos14.py --resume vit_adaptive_1100 --round 1400 &&
deepspeed --include="localhost:1,3" --master_port 12109 train_adaptive_thumos14.py --resume vit_adaptive_1400 --round 1700 &&
deepspeed --include="localhost:1,3" --master_port 12109 train_adaptive_thumos14.py --resume vit_adaptive_1700 --round 2000 &&
deepspeed --include="localhost:1,3" --master_port 12109 train_adaptive_thumos14.py --resume vit_adaptive_2000 --round 2300