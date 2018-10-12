#!/bin/bash
set -e
set -x
EXEC_DIR="mouth48"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=models/mouth48_1011/1011_48_iter_120000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
