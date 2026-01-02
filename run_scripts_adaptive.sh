#!/bin/bash
source /data/xiaodan8/anaconda3/bin/activate parsing2

# cp -R /data3/xiaodan8/actionformer4_1/ckpt/finegym_i3d_uniform_ds/vit_uniform_1000 /data3/xiaodan8/actionformer4_1/ckpt/finegym_i3d_adaptive_ds/vit_adaptive_1000 &&
# # deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_1000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_1000 --round 2000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_2000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_2000 --round 3000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_3000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_3000 --round 4000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_4000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_4000 --round 5000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_5000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_5000 --round 6000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_6000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_6000 --round 7000 &&
# deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_7000 &&
deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_7000 --round 8000 &&
deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_8000 &&
deepspeed --include="localhost:2,3" --master_port 12108 train_adaptive.py --resume vit_adaptive_8000 --round 9000 &&
deepspeed --include="localhost:2,3" --master_port 12108 eval_shard.py --config ./configs/finegym_i3d_adaptive.yaml --ckpt ./ckpt/finegym_i3d_adaptive_ds --filename vit_adaptive_9000