#!/bin/bash
set -e
set -x
EXEC_DIR="mouthSmall"
export PYTHONPATH=/Users/momo/wkspace/caffe_space/detection/caffe/python:$PYTHONPATH
export PYTHONPATH=/Users/momo/wkspace/caffe_space/detection/caffe/examples/hand_reg/${EXEC_DIR}:$PYTHONPATH
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_reg/${EXEC_DIR}/solver.prototxt \
  --weights=models/mouth/1008f_iter_10000.caffemodel \
  2>&1 | tee examples/hand_reg/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
