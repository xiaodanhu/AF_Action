#!/bin/bash
source /data/xiaodan8/anaconda3/bin/activate parsing2

deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --round 2500 &&
deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --resume vit_uniform_2500 --round 5000 &&
deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --resume vit_uniform_5000 --round 7500 &&
deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --resume vit_uniform_7500 --round 10000 &&
deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --resume vit_uniform_10000 --round 12500 &&
deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --resume vit_uniform_12500 --round 15000 &&
deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --resume vit_uniform_15000 --round 17500 &&
deepspeed --include="localhost:1,3" --master_port 12166 train_uniform.py --resume vit_uniform_17500 --round 20000