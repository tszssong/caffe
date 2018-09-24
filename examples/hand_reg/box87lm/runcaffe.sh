#!/bin/bash
set -e
set -x
EXEC_DIR="box87lm"
export PYTHONPATH=/Users/momo/wkspace/caffe_space/detection/caffe/python:$PYTHONPATH
export PYTHONPATH=/Users/momo/wkspace/caffe_space/detection/caffe/examples/hand_reg/${EXEC_DIR}:$PYTHONPATH
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_reg/${EXEC_DIR}/solver.prototxt \
  --weights=models/box87lm/0920flip_iter_30000.caffemodel \
  2>&1 | tee examples/hand_reg/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
