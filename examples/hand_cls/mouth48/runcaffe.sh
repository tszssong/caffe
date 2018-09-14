#!/bin/bash
set -e
set -x
EXEC_DIR="mouth48"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=models/cls48/hand_gesture_cls_v3.1_180907.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
