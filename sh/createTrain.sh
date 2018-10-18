#!/usr/bin/env sh
set -e
EXAMPLE=sh
DATA=data/64data/
TOOLS=build/tools

TRAIN_DATA_ROOT=/Users/momo/wkspace/caffe_space/caffe/data/64data/

RESIZE_HEIGHT=64
RESIZE_WIDTH=64

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
    $DATA/1018bg64/up_1018bg64.txt\
    $DATA/1018bg64_lmdb
echo "Done."
