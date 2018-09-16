#!/bin/bash
set -e
set -x
EXEC_DIR="bare48"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=examples/hand_cls/${EXEC_DIR}/180824_iter_700000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
