#!/bin/bash
set -e
set -x
EXEC_DIR="mouth64"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
