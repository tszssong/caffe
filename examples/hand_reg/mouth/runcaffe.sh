#!/bin/bash
set -e
set -x
EXEC_DIR="mouth"
export PYTHONPATH=/nfs/zhengmeisong/wkspace/gesture/caffe/python:$PYTHONPATH
export PYTHONPATH=/nfs/zhengmeisong/wkspace/gesture/caffe/examples/hand_reg/${EXEC_DIR}:$PYTHONPATH
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_reg/${EXEC_DIR}/solver.prototxt \
  --weights=models/mouth/1008f_iter_10000.caffemodel \
  2>&1 | tee examples/hand_reg/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
