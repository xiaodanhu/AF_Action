#!/bin/bash

# finegym with I3D features
python ./train.py --output final 2>&1 | tee ./ckpt/finegym_log.txt
# python ./eval.py 2>&1 | tee ./ckpt/finegym_results.txt