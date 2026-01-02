#!/bin/bash
source /data/xiaodan8/anaconda3/bin/activate parsing2

# cp -R /data3/xiaodan8/actionformer4_1/ckpt/finegym_i3d_uniform_ds/vit_uniform_2500 /data3/xiaodan8/actionformer4_1/ckpt/finegym_i3d_adaptive_v2_ds_v2/vit_adaptive_2500_v2 &&
# deepspeed --include="localhost:2,3" --master_port 12366 train_uniform.py --resume vit_uniform_2500 --round 2500 &&
# cp /data3/xiaodan8/actionformer4_1/output/finegym/logits_uniform_2500.pt /data3/xiaodan8/actionformer4_1/output/finegym/logits_adaptive_2500.pt &&
# cp /data3/xiaodan8/actionformer4_1/output/finegym/pred_vit_uniform_2500_ap.npy /data3/xiaodan8/actionformer4_1/output/finegym/pred_vit_adaptive_2500_ap.npy &&
# cp /data3/xiaodan8/actionformer4_1/output/finegym/pred_vit_uniform_2500.json /data3/xiaodan8/actionformer4_1/output/finegym/pred_vit_adaptive_2500.json &&
# deepspeed --include="localhost:2,3" --master_port 12366 train_adaptive_v2.py --resume vit_adaptive_2500_v2 --round 5000 &&
# deepspeed --include="localhost:2,3" --master_port 12366 train_adaptive_v2.py --resume vit_adaptive_5000_v2 --round 7500 &&
# deepspeed --include="localhost:2,3" --master_port 12366 train_adaptive_v2.py --resume vit_adaptive_7500_v2 --round 10000 &&
# deepspeed --include="localhost:2,3" --master_port 12366 train_adaptive_v2.py --resume vit_adaptive_10000_v2 --round 12500 &&
# deepspeed --include="localhost:2,3" --master_port 12366 train_adaptive_v2.py --resume vit_adaptive_12500_v2 --round 15000 &&
deepspeed --include="localhost:2,3" --master_port 12366 train_adaptive_v2.py --resume vit_adaptive_15000_v2 --round 17500 &&
deepspeed --include="localhost:2,3" --master_port 12366 train_adaptive_v2.py --resume vit_adaptive_17500_v2 --round 20000