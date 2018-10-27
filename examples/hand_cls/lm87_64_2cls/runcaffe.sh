#!/bin/bash
set -e
set -x
EXEC_DIR="lm87_64_2cls"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=/Users/momo/wkspace/caffe_space/detection/caffe/models/fromAli/0927data_b256_iter_1000000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
.                                      
