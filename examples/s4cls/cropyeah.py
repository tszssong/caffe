import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from utils import IOU, overlapSelf
from image_argument import flipAug, rotAug
cropSize = 48
THpos = 0.3
ScaleS = 1.0
ScaleB = 1.5
Ratio = 'R'  # crop recording the width&height ratio
Shift = 0.3
RotD = 25
from_dir = "/Volumes/song/gestureDatabyName/yeah-img/"
to_dir = "/Users/momo/wkspace/caffe_space/caffe/data/clsData/"
anno_file = "/Users/momo/wkspace/caffe_space/caffe/examples/s4cls/gt/2-yearh-train.txt"
clslists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fheart', 'ok', 'call', 'rock', 'big_v','otherhand','fist','ILU']
annofileName = anno_file.split('.')[0].split('/')[-1]
print annofileName
clsname = annofileName.split('-')[-2]
cls_idx = clslists.index(clsname)

N_RESIZE = 2
N_ROT = 4
date = "_0915"

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
    if p_idx > 20000:
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

        img, f_bbox = rotAug(image, boxes, rot_d)

        f_boxes = np.array(f_bbox, dtype=np.float32).reshape(-1, 4)

        oriH, oriW, oriC = img.shape
        height, width, channel = img.shape

        for box_idx in xrange(f_boxes.shape[0]):

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
            if( (b - g) < 10 and abs( b - r) < 10):
                continue

            for i in range(N_RESIZE):
                nw = npr.randint(np.ceil(rw * ScaleS), np.ceil(ScaleB * rw))
                if Ratio == 'R':
                    ratio = float(rh) / float(rw)
                    nh = int(ratio * nw)
                else:
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

                if nx2 > width or ny2 > height or nx1 < 0 or ny1 < 0:
                    continue
                if nx2 < rx2 - 1 or nx1 > rx1 + 1 or ny2 < ry2 - 1 or ny1 > ry1 + 1:
                    continue


                ncropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                crop_box = np.array([nx1, ny1, nx2, ny2])

                overlap_flag = 0
                if nbox > 1:
                    otherboxes = np.array([])
                    for otherbox_idx in xrange(f_boxes.shape[0]):
                        if not box_idx == otherbox_idx:
                            iou = IOU(crop_box, f_boxes[otherbox_idx])
                            # otherboxes = np.append(otherboxes, f_boxes[otherbox_idx])
                            if iou > 0.1:
                                overlap_flag = 1


                if overlap_flag == 1:
                    continue

                Iou = overlapSelf(crop_box, box)
                if np.max(Iou) < THpos:
                    continue

                nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
                grey = 0.11 * b + 0.59 * r + 0.3 * g
                if (grey < 70):
                    # print im_path,"brightness not good!"
                    continue
                    # cv2.imshow("grey", cropped_im)
                    # cv2.imshow("resized", nresized_im)
                    # cv2.waitKey()
                #drint im_path, "std:", np.std(nresized_im[:, :, 0]), np.std(nresized_im[:, :, 1]), np.std(nresized_im[:, :, 2])

                # if (abs(np.mean(nresized_im[:,:,0]) - np.mean(nresized_im[:,:,1]))<10 and abs(np.mean(nresized_im[:,:,0]) -  np.mean(nresized_im[:,:,2]))<10):
                #     continue

                box_ = box.reshape(1, -1)

                filename = '/'+im_path.split('.')[0]+'_'+str(p_idx)+'.jpg'

                save_file = os.path.join(save_dir + filename)
                fw.write(save_dir + filename + ' ' + str(cls_idx) + '\n')

                cv2.imwrite(to_dir + save_file, nresized_im)

                p_idx += 1
            box_idx += 1

fw.close()

