#!/bin/bash
set -e
set -x
EXEC_DIR="alimouth"
export PYTHONPATH=/nfs/zhengmeisong/wkspace/gesture/caffe/python:$PYTHONPATH
export PYTHONPATH=/nfs/zhengmeisong/wkspace/gesture/caffe/examples/hand_reg/${EXEC_DIR}:$PYTHONPATH
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_reg/${EXEC_DIR}/solver.prototxt \
  --weights=models/mouth/1026f1025_iter_1500000.caffemodel \
  --gpu 5 2>&1 | tee examples/hand_reg/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
