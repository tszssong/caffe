#!/usr/bin/env sh
set -e
EXAMPLE=sh
DATA=data/64data/
TOOLS=build/tools

TRAIN_DATA_ROOT=/Users/momo/wkspace/caffe_space/caffe/data/64data/

RESIZE_HEIGHT=48
RESIZE_WIDTH=48


if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/up_1015addbgTrain.txt\
    $DATA/up_1015addbgTrain48_lmdb

echo "Done."
