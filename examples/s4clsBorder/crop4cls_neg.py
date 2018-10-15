import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from utils import IOU, overlapSelf
from cropNegs import cropNeg
from image_argument import flipAug, rotAug

cropSize = 64
N_CROP = 3

# from_dir = "/Volumes/song/gestureDatabyName/1-heart-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/1-heart-xml.txt"
# from_dir = "/Volumes/song/gestureDatabyName/2-yearh-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/2-yearh-xml.txt"
# from_dir = "/Volumes/song/gestureDatabyName/3-one-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/3-one-train.txt"
# from_dir = "/Volumes/song/gestureTight4Reg/Tight5-notali2-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/5-five-Tightnoali2.txt"
# from_dir = "/Volumes/song/gestureDatabyName/5-five-VggMomo-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/5-five-VggMomo.txt"

from_dir = "/Volumes/song/gestureDatabyName/11-rock-img/"
anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/11-rock-xml.txt"
# from_dir = "/Volumes/song/gestureDatabyName/7-zan-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/7-zan-train.txt"
to_dir = "/Users/momo/wkspace/caffe_space/caffe/data/64data/"

annofileName = anno_file.split('.')[0].split('/')[-1]
print annofileName
gen_mode = annofileName.split('-')[-1] #train/test
clsname = annofileName.split('-')[-2]
cls_idx = 0
date = "_1012"
txt_name = '0-' + annofileName + '_neg' + date
save_dir = '0-' + annofileName + '_neg' + date
print gen_mode, clsname, cls_idx
print txt_name, save_dir

if not os.path.exists(to_dir+save_dir):
    os.mkdir(to_dir+save_dir)

fw = open(to_dir + '/Txts/'+txt_name + '.txt', 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()

num = len(annotations)
print "%d pics in total" % num, "NCROP:", N_CROP
p_idx = 0  # positive
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    #print im_path
    nbox = int(annotation[1])
    bbox = map(float, annotation[3:])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    image = cv2.imread(os.path.join(from_dir, im_path))
    idx += 1
    if idx % 100 == 0:
        print "%s images done, pos: %s "%(idx, p_idx)
    height, width, channel = image.shape

    neg_num = 0
    loopNeg = 0
    for pic_idx in range(N_CROP):
        nresized_im, ret = cropNeg(image, boxes, cropSize, pic_idx%4);
        # nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
        if ret:
            filename = '/' + im_path.split('.')[0] + '_' + str(p_idx) + '.jpg'
            save_file = os.path.join(save_dir + filename)
            fw.write(save_dir + filename + ' ' + str(cls_idx) + '\n')
            cv2.imwrite(to_dir + save_file, nresized_im)
            p_idx += 1

fw.close()

