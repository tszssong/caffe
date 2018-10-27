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
ScaleS = 2.2
ScaleB = 3.2
Shift =  0.5
maxNum = 50000
im_dir = "/nfs/zhengmeisong/wkspace/gesture/VOCdevkit/VOC2007/JPEGImages/"
anno_file = "gt/T-5-20180724_hebing.txt"
#anno_file = "gt/T-5-201807027_disturb.txt"
#anno_file = "gt/T-5-grab.txt"
#anno_file = "gt/T-5-good.txt"
#anno_file = "gt/T-2-yeah-total.txt"
#anno_file = "gt/T-3-one-total.txt"
#anno_file = "gt/T-7-zan-total.txt"
#anno_file = "gt/T-8-fheart-total.txt"
#anno_file = "gt/T-9-ok-total.txt"
#anno_file = "gt/T-10-call-total.txt"
#anno_file = "gt/T-11-rock-total.txt"
#anno_file = "gt/T-12-big_v-total.txt"
#anno_file = "gt/T_13_fist.txt"
anno_file = "gt/T-15-palm-total.txt"
# anno_file = "gt/Tight_green_nofist.txt"
to_dir = "/nfs/zhengmeisong/wkspace/gesture/caffe/data/regData/1027data/"
annofileName = anno_file.split('.')[0].split('/')[-1]
save_name = annofileName +'_' + str(cropSize)+ 'S'+ str(ScaleS).split('.')[0] + str(ScaleS).split('.')[1] + str(int(ScaleB * 10)) + '_' + str(int(Shift * 10)) +'_' +str(flipRange)+ paddingMode
save_dir = save_name

if not os.path.exists(to_dir+save_dir):
    os.mkdir(to_dir+save_dir)
fw = open(os.path.join(to_dir+'Txts/', save_dir+'.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
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

