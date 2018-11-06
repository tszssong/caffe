import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from utils import IOU, overlapSelf, overlapingOtherBox
from bbox_transform import validBox, crop4reg
from image_process import fliterDim, crop_image
from image_argument import flipAug
paddingMode = 'black'
OutAllowed = 10   # 10 pixels allowed to go out of gt
cropSize = 64
flipRange = 2  #flip params: 1-ori\horizontal
ScaleS = 1.0
ScaleB = 2.8
Shift =  0.5
maxNum = 10000
im_dir = "/Volumes/song/handgesture5_48G/Tight_ali2_five_train-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_5_ali2.txt"

im_dir = "/Volumes/song/gestureTight4Reg/Tight5-notali2-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_5_hebing.txt"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_5_notali2_ali2grab.txt"

im_dir = "/Volumes/song/gestureDatabyName/2-yearh-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_2_yeah.txt"
im_dir = "/Volumes/song/gestureDatabyName/3-one-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_3_one.txt"

im_dir = "/Volumes/song/gestureDatabyName/7-zan-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_7_zan.txt"
im_dir = "/Volumes/song/gestureDatabyName/8-fingerheart-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_8_fheart.txt"
im_dir = "/Volumes/song/handgTight_56G/T_9_ok1ali2-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_9_ok1ali2-xml.txt"
im_dir = "/Volumes/song/handgTight_56G/T_10_ali1call-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_10_ali1call-xml.txt"
im_dir = "/Volumes/song/handgTight_56G/T_11_ali1rock1-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_11_ali1rock1-xml.txt"
im_dir = "/Volumes/song/gestureDatabyName/12-big_v-img//"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T-12-big_v-total.txt"

im_dir = "/Volumes/song/gestureDatabyName/13-fist-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_13_fist.txt"

im_dir = "/Volumes/song/gestureTight4Reg/Tight-palm-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_palm.txt"

im_dir = "/Volumes/song/gestureTight4Reg/Tight_green_nofist-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/Tight_green_nofist.txt"

im_dir = "/Volumes/song/Tight_pink/20181026_one_pink-img/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_3_one-pink.txt"

# im_dir = "/Volumes/song/Tight_pink/20181025_five-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_5_five-pink.txt"

# im_dir = "/Volumes/song/Tight_pink/20181026_fist_pink-img"
# anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_13_fist-pink.txt"

to_dir = "/Users/momo/wkspace/caffe_space/detection/caffe/data/1103reg64/"
annofileName = anno_file.split('.')[0].split('/')[-1]
save_name = annofileName +'_' + str(cropSize)+ 'S'+ str(ScaleS).split('.')[0] + str(ScaleS).split('.')[1] + str(int(ScaleB * 10)) + '_' + str(int(Shift * 10)) +'_' +str(flipRange)+ paddingMode
save_dir = save_name

if not os.path.exists(to_dir+save_dir):
    os.mkdir(to_dir+save_dir)
fw = open(os.path.join(to_dir+'Txts/', save_dir+'.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print anno_file.split('/')[-1]
print "%d pics in total" % num, "%d needed" % maxNum
croped_pic_idx = 0  # positive
ori_pic_idx = 0

while(croped_pic_idx<maxNum):
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        #print im_path
        nbox = int(annotation[1])

        if nbox>2:
            continue
        if croped_pic_idx >= maxNum:
            break

        bbox = map(float, annotation[3:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        image = cv2.imread(os.path.join(im_dir, im_path))
        ori_pic_idx += 1
        if ori_pic_idx % 100 == 0:
            print "%s images done, pos: %s"%(ori_pic_idx, croped_pic_idx)

        height, width, channel = image.shape

        flip_arg = np.random.randint(0,flipRange)                             #randint(0,5) -- int of 0/1/2/3/4
        img, f_bbox = flipAug(image, boxes, flip_arg)                         #take attention! need to deep copy new img)
        f_boxes = np.array(f_bbox, dtype=np.float32).reshape(-1, 4)
        height, width, channel = img.shape

        for box_idx in xrange(f_boxes.shape[0]):
            box = f_boxes[box_idx]
            if validBox(box, width, height) == False:
                continue
            cropped_im = img[int(box[1]): int(box[3]), int(box[0]): int(box[2]), :]
            if fliterDim(cropped_im) == False:
                continue

            # crop_box, reg_coord = crop4reg(box, ScaleS, ScaleB, Shift, OutAllowed)
            crop_box, reg_coord = crop4reg(box_idx, f_boxes, ScaleS, ScaleB, Shift, OutAllowed)
            if not crop_box.size == 4:
                continue

            # if nbox > 1:
            #     if overlapingOtherBox(crop_box, box_idx, f_boxes):
            #         continue

            ncropped_im = crop_image(img, crop_box, paddingMode)
            nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
            # nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize), interpolation=cv2.INTER_LINEAR)
            box_ = box.reshape(1, -1)
            filename = "/" + str(croped_pic_idx) + '_' +  im_path.split('.')[0] + '.jpg'
            save_file = os.path.join(save_dir + filename)
            fw.write(save_dir + filename + ' 1 %.5f %.5f %.5f %.5f\n' % (reg_coord[0], reg_coord[1], reg_coord[2], reg_coord[3]))
            cv2.imwrite(to_dir + save_file, nresized_im)
            croped_pic_idx += 1
            box_idx += 1

print croped_pic_idx, "images croped"
fw.close()

