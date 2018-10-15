#!/bin/bash
set -e
set -x
EXEC_DIR="lm87cls"
export PYTHONPATH=/Users/momo/wkspace/caffe_space/caffe/python:$PYTHONPATH
export PYTHONPATH=/Users/momo/wkspace/caffe_space/caffe/examples/hand_reg/${EXEC_DIR}:$PYTHONPATH
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=/Users/momo/wkspace/caffe_space/detection/caffe/models/lm87_64/0922allfb512_iter_120000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
