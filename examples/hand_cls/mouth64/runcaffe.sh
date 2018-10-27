#!/bin/bash
set -e
set -x
EXEC_DIR="mouth64"
TOOLS=./build/tools
$TOOLS/caffe train \
  --solver=examples/hand_cls/${EXEC_DIR}/solver.prototxt \
  --weights=models/fromAli/mouth64bn/1018addbg_1012f_iter_1000000.caffemodel \
  2>&1 | tee examples/hand_cls/${EXEC_DIR}/log/`date +'%Y-%m-%d_%H-%M-%S'`.log
