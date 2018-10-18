import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from utils import IOU, overlapSelf, overlapingOtherBox
from bbox_transform import crop4cls, validBox
from image_process import  crop_image, fliterDim
from image_argument import flipAug, rotAug
OutAllowed = 10
cropSize = 64
ScaleS = 1.0
ScaleB = 2.0
Shift = 1.5
paddingMode = 'black'
N_RESIZE = 4
N_ROT = 1
date = "_1017"
# from_dir = "/Volumes/song/gestureTight4Reg/Tight5-notali2-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/5-five-Tightnoali2.txt"

# from_dir = "/Volumes/song/handgesture5_48G/Tight_ali2_five_train-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/5-five-TPickedali2.txt"

from_dir = "/Volumes/song/gestureDatabyName/5-five-VggMomo-img/"
anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/5-five-VggMomo.txt"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/5-five-gzjp41.txt"

# from_dir = "/Volumes/song/handgesture5_48G/ali1five_sy-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/5-five-syali1.txt"
to_dir = "/Users/momo/wkspace/caffe_space/caffe/data/48Test/"
clslists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fingerheart', 'ok', 'call', 'rock', 'big_v','fist','palm', 'namaste', 'two_together', 'thumb_down']
RotDlists = [ 0,       5,       30,    10,         5,    110,         5,    10,            10,   30,     30,     30,      30,    30,     5,         5,              5,     5]
annofileName = anno_file.split('.')[0].split('/')[-1]
print annofileName

save_name = annofileName +'_' + str(cropSize)+ 'S'+ str(ScaleS).split('.')[0] + str(ScaleS).split('.')[1] + str(int(ScaleB * 10)) + date
save_dir = save_name
txt_name = save_name

clsname = annofileName.split('-')[-2]
print txt_name, save_dir

if not os.path.exists(to_dir+save_dir):
    os.mkdir(to_dir+save_dir)
fw = open(to_dir + '/Txts/' + txt_name + '.txt', 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)

print "%d pics in total" % num, "NROT:", N_ROT,"N_RESIZE:", N_RESIZE
p_idx = 0  # positive
idx = 0
box_idx = 0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    nbox = int(annotation[1])

    if nbox>2:
        continue
    # if len(annotation[3:]) > 4:
    #     annotation.pop(7)

    objname = annotation[2]
    if (objname == 'two_together' or objname == 'thumb_down' or objname == 'palm' or objname == 'namaste'):
        continue
    cls_idx = clslists.index(objname)

    RotD = RotDlists[cls_idx]
    bbox = map(float, annotation[3:])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    image = cv2.imread(os.path.join(from_dir, im_path))

    idx += 1
    if idx % 100 == 0:
        print "%s images done, pos: %s" % (idx, p_idx)

    height, width, channel = image.shape

    for pic_idx in range(N_ROT):

        rot_d = np.random.randint(-RotD, RotD)
        img, f_bbox, f_flag = rotAug(image, boxes, rot_d)
        f_boxes = np.array(f_bbox, dtype=np.float32).reshape(-1, 4)
        height, width, channel = img.shape

        for box_idx in xrange(f_boxes.shape[0]):

            if f_flag[box_idx] == False:      #skip boxes outside image after Rot
                continue
            box = f_boxes[box_idx]
            if validBox(box, width, height) == False:
                continue
            # cropped_im = img[int(ry1): int(ry2), int(rx1): int(rx2), :]
            cropped_im = img[int(box[1]): int(box[3]), int(box[0]): int(box[2]), :]
            if fliterDim(cropped_im) == False:
                continue

            for i in range(N_RESIZE):
                crop_box = crop4cls(box, ScaleS, ScaleB, Shift,OutAllowed)
                if not crop_box.size == 4:
                    continue

                if nbox>1:
                    if overlapingOtherBox(crop_box, box_idx, f_boxes):
                        continue

                ncropped_im = crop_image(img, crop_box, paddingMode)
                nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
                filename = '/'+str(cls_idx)+'_'+str(p_idx)+'_'+im_path.split('.')[0]+'.jpg'
                save_file = os.path.join(save_dir + filename)
                fw.write(save_dir + filename + ' ' + str(cls_idx) + '\n')
                cv2.imwrite(to_dir + save_file, nresized_im)
                p_idx += 1
            box_idx += 1
fw.close()

