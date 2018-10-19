#!/bin/bash
set -e
set -x
EXEC_DIR="lm87bcls"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=/Users/momo/wkspace/caffe_space/caffe/models/lm87bcls/2cls_iter_10000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
