import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from utils import IOU, overlapSelf
from image_argument import flipAug, rotAug
cropSize = 64
ScaleS = 1.0
ScaleB = 2.0
Shift = 1.5
RotD = 30
# from_dir = "/Volumes/song/gestureDatabyName/2-yearh-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/2-yearh-xml.txt"
# from_dir = "/Volumes/song/gestureDatabyName/9-ok-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/9-ok-xml.txt"
# from_dir = "/Volumes/song/gestureDatabyName/10-call-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/10-call-xml.txt"
# from_dir = "/Volumes/song/gestureDatabyName/11-rock-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/11-rock-left.txt"
# from_dir = "/Volumes/song/gestureDatabyName/12-big_v-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/12-big_v-xml.txt"
from_dir = "/Volumes/song/gestureDatabyName/13-fist-img/"
anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4clsBorder/gt/13-fist-xml.txt"
to_dir = "/Users/momo/wkspace/caffe_space/caffe/data/64data/"
clslists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fingerheart', 'ok', 'call', 'rock', 'big_v', 'fist', 'other']
annofileName = anno_file.split('.')[0].split('/')[-1]
print annofileName
clsname = annofileName.split('-')[-2]
cls_idx = clslists.index(clsname)

N_RESIZE = 10
N_ROT = 5
date = "_1014"

save_name = annofileName +'_' + str(cropSize)+ 'R'+str(RotD) +'S'+ str(ScaleS).split('.')[0] + str(ScaleS).split('.')[1] + str(int(ScaleB * 10)) + date
save_dir = save_name + '_1'
txt_name = save_name + '_1'

print clsname, cls_idx
print txt_name, save_dir

if not os.path.exists(to_dir+save_dir):
    os.mkdir(to_dir+save_dir)
fw = open(to_dir + '/Txts/' + txt_name + '.txt', 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()

num = len(annotations)

print "%d pics in total" % num, "NROT:", N_ROT,"N_RESIZE:", N_RESIZE
p_idx = 0  # positive
d_idx = 1  # dict idx
idx = 0
box_idx = 0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    nbox = int(annotation[1])
    if nbox>2:
        continue
    objname = annotation[2]

    if not objname == clsname:
        # print im_path ," with class :", objname
        continue
    bbox = map(float, annotation[3:])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

    image = cv2.imread(os.path.join(from_dir, im_path))

    idx += 1
    if idx % 100 == 0:
        print "%s images done, pos: %s" % (idx, p_idx)
    if p_idx > 100000:
        p_idx = 0
        d_idx += 1
        txt_name = save_name + '_' + str(d_idx)
        save_dir = save_name + '_' + str(d_idx)
        if not os.path.exists(to_dir + save_dir):
            os.mkdir(to_dir + save_dir)
        fw.close()
        fw = open(to_dir + '/Txts/' + txt_name + '.txt', 'w')

    height, width, channel = image.shape



    for pic_idx in range(N_ROT):
        rot_d = np.random.randint(-RotD, RotD)

        img, f_bbox, f_flag = rotAug(image, boxes, rot_d)

        f_boxes = np.array(f_bbox, dtype=np.float32).reshape(-1, 4)

        oriH, oriW, oriC = img.shape
        height, width, channel = img.shape

        for box_idx in xrange(f_boxes.shape[0]):
            if f_flag[box_idx] == False:      #skip boxes outside image aftre Rot
                continue

            box = f_boxes[box_idx]

            x1, y1, x2, y2 = box
            rx1, ry1, rx2, ry2 = box
            if rx1>=rx2 or ry1>=ry2:
                continue
            
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if max(w, h) < 60 or x1 < 0 or y1 < 0:
                continue
            rw = rx2 - rx1
            rh = ry2 - ry1
            rcx = rx1 + rw / 2
            rcy = ry1 + rh / 2
            if x1 > x2 or y1 > y2:
                print ":", x1, y1, x2, y2, "-", width, height
            if x2 > width or y2 > height or x1 < 0 or y1 < 0:
                continue

            cropped_im = img[int(ry1): int(ry2), int(rx1): int(rx2), :]
            b = np.mean(cropped_im[:, :, 0])
            g = np.mean(cropped_im[:, :, 1])
            r = np.mean(cropped_im[:, :, 2])
            if ((b - g) < 10 and abs(b - r) < 10):  # blur img
                continue
            grey = 0.11 * b + 0.59 * r + 0.3 * g
            if (grey < 70):  # dark img
                continue

            for i in range(N_RESIZE):
                # maxWH = np.max((rw, rh))
                # enlargeS = npr.randint(np.ceil(maxWH * ScaleS), np.ceil(ScaleB * maxWH))
                # delta_x = npr.randint(-float(rw) * Shift, float(rw) * Shift)
                # delta_y = npr.randint(-float(rh) * Shift, float(rh) * Shift)

                nw = npr.randint(np.ceil(rw * ScaleS), np.ceil(ScaleB * rw))
                nh = nw
                if i == 0 and nw < w * 1.5:
                    delta_x = npr.randint(-int(float(rw) * 0.2), int( float(rw) * 0.2 ) + 1)
                    delta_y = npr.randint(-int(float(rh) * 0.2), int( float(rh) * 0.2 ) + 1)
                else:
                    delta_x = npr.randint(-int(float(rw) * Shift), int(float(rw) * Shift+1))
                    delta_y = npr.randint(-int(float(rh) * Shift), int(float(rh) * Shift+1))

                ncx = rcx + delta_x
                ncy = rcy + delta_y

                nx1 = ncx - nw / 2
                ny1 = ncy - nh / 2
                nx2 = ncx + nw / 2
                ny2 = ncy + nh / 2

                if nx2 < rx2 - 10 or nx1 > rx1 + 10 or ny2 < ry2 - 10 or ny1 > ry1 + 10:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #iou 2 include
                overlap_flag = 0
                if nbox > 1:
                    otherboxes = np.array([])
                    for otherbox_idx in xrange(f_boxes.shape[0]):
                        if not box_idx == otherbox_idx:
                            iou = IOU(crop_box, f_boxes[otherbox_idx])
                            # otherboxes = np.append(otherboxes, f_boxes[otherbox_idx])
                            if iou > 0.01:
                                overlap_flag = 1

                if overlap_flag == 1:
                    continue

                right_x = 0
                left_x = 0
                top_y = 0
                down_y = 0
                constant = copy.deepcopy(img)
                if nx2 > width or ny2 > height or nx1 < 0 or ny1 < 0:
                    if nx2 > width:
                        right_x = nx2 - width
                    if ny2 > height:
                        down_y = ny2 - height
                    if nx1 < 0:
                        left_x = 0 - nx1
                    if ny1 < 0:
                        top_y = 0 - ny1
                    black = [0, 0, 0]
                    # print "edge:", top_y, down_y, left_x, right_x
                    constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_CONSTANT, value = black );
                    # constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_REPLICATE);

                ncropped_im = constant[int(ny1 + top_y):int(ny2 + top_y), int(nx1 + left_x):int(nx2 + left_x), :]
                nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)

                box_ = box.reshape(1, -1)

                filename = '/'+str(p_idx)+'_'+im_path.split('.')[0]+'.jpg'

                save_file = os.path.join(save_dir + filename)
                fw.write(save_dir + filename + ' ' + str(cls_idx) + '\n')

                cv2.imwrite(to_dir + save_file, nresized_im)

                p_idx += 1
            box_idx += 1

fw.close()

