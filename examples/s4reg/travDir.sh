#!/bin/bash
set -e 
set -x
for file in `ls gt_total`
do
python ali_crop400.py $file &
#python ali_cropLists.py 96 $file   &
done

