#!/bin/bash
set -e
set -x
EXEC_DIR="mouth48"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=examples/hand_cls/${EXEC_DIR}/models/fist13cls_iter_370000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
