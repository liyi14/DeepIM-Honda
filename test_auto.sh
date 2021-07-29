#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHONPATH=$PYTHONPATH:$PWD
python ./tools/demo_test.py \
  --gpu -1 \
  --trainset train@curiosity \
  --testset train@curiosity \
  --class_ref none \
  --class_test none \
  --cfg experiments/cfgs/curiosity_fnot_rgbd_large_scale_v10_baseline.yml \
  --phase REF \
  --vis
