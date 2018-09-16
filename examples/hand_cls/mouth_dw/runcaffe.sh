#!/bin/bash
set -e
set -x
EXEC_DIR="mouth_dw"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=models/mouth_dw0915/13cls_iter_420000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
