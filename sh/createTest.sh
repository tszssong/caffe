#!/usr/bin/env sh
set -e
DATA=data/1027cls64/
#TODATA=/Volumes/song/2mac/cls64/
TOOLS=build/tools

VAL_DATA_ROOT=/Users/momo/wkspace/caffe_space/caffe/

RESIZE_HEIGHT=64
RESIZE_WIDTH=64
#RESIZE_HEIGHT=48
#RESIZE_WIDTH=48

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    $DATA/1027_test64_lmdb

echo "Done."
