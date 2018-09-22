import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from utils import IOU, overlapSelf
from image_argument import  flipAug
cropSize = 64
N_FLIP = 10
N_RESIZE = 40
THneg = 0.001
THpos = 0.3
ScaleFacetors = np.array([10,10,5,5])
ScaleS = 1.0
ScaleB = 2.9
Ratio = 'R'      #crop recording the width&height ratio
Shift = 2.7

# anno_file = "//Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/5-ali2five.txt"
# im_dir = "/Volumes/song/handgesture5/Tight_ali2_five_train-img/"
# anno_file = "//Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/Tight5-notali2.txt"
# im_dir = "/Volumes/song/handgesture5/Tight5-notali2-img/"
anno_file = "//Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/Tight_20180724_five_hebing.txt"
im_dir = "/Volumes/song/handgesture5/Tight/Tight_20180724_five_hebing-img/"
to_dir = "/Users/momo/wkspace/caffe_space/detection/caffe/data/reg0921/"
annofileName = anno_file.split('.')[0].split('/')[-1]
suffix = '_hflip'
save_name = annofileName +'_' + str(cropSize)+ 'S'+ str(ScaleS).split('.')[0] + str(ScaleS).split('.')[1] + str(int(ScaleB * 10)) + suffix
save_dir = save_name + '_1'

if not os.path.exists(to_dir+save_dir):
    os.mkdir(to_dir+save_dir)
fw = open(os.path.join(to_dir+'Txts/', save_dir+'.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num
p_idx = 0  # positive
idx = 0
box_idx = 0
d_idx = 1  # dict idx
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    #print im_path
    nbox = int(annotation[1])

    bbox = map(float, annotation[3:])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

    image = cv2.imread(os.path.join(im_dir, im_path))

    idx += 1
    if idx % 100 == 0:
        print "%s images done, pos: %s"%(idx, p_idx)

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

    neg_num = 0
    loopNeg = 0

    for pic_idx in range(N_FLIP):
        # if pic_idx%2 == 0:
        #     flip_arg = 0
        # else:
        #     flip_arg = 5
        flip_arg = pic_idx%5
        if flip_arg == 1 or flip_arg ==2:
            flip_arg = 0

        img, f_bbox = flipAug(image, boxes, flip_arg)                 #take attention! need to deep copy new img
        f_boxes = np.array(f_bbox, dtype=np.float32).reshape(-1, 4)

        oriH, oriW, oriC = img.shape
        height, width, channel = img.shape

        for box_idx in xrange(f_boxes.shape[0]):

            box = f_boxes[box_idx]
            x1, y1, x2, y2 = box
            rx1, ry1, rx2, ry2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            rw = rx2 - rx1
            rh = ry2 - ry1
            rcx = rx1 + rw / 2
            rcy = ry1 + rh / 2
            if x1>x2 or y1>y2:
                print ":",x1,y1,x2,y2,"-",width,height
            if x2 > width or y2 > height or x1 < 0 or y1 < 0 :
                continue

            cropped_im = img[int(ry1) : int(ry2), int(rx1) : int(rx2), :]
            b = np.mean(cropped_im[:, :, 0])
            g = np.mean(cropped_im[:, :, 1])
            r = np.mean(cropped_im[:, :, 2])
            if ((b - g) < 10 and abs(b - r) < 10):
                continue
            grey = 0.11 * b + 0.59 * r + 0.3 * g
            if (grey < 80):
                continue
            for i in range(N_RESIZE):
                nw = npr.randint(np.ceil(rw*ScaleS), np.ceil(ScaleB * rw))
                if Ratio == 'R':
                    ratio = float(rh)/float(rw)
                    nh = int(ratio*nw)
                else:
                    nh = nw
                if i==0 and nw < w*1.5:
                    delta_x = npr.randint(-float(rw)*0.2, float(rw)*0.2)
                    delta_y = npr.randint(-float(rh)*0.2, float(rh)*0.2)
                else:
                    delta_x = npr.randint(-float(rw)*Shift, float(rw)*Shift)
                    delta_y = npr.randint(-float(rh)*Shift, float(rh)*Shift)

                ncx = rcx + delta_x
                ncy = rcy + delta_y
                
                nx1 = ncx - nw/2
                ny1 = ncy - nh/2
                nx2 = ncx + nw/2
                ny2 = ncy + nh/2

                if nx2 > width or ny2 > height or nx1 < 0 or ny1 < 0 :
                    continue
                if nx2 < rx2-rw/5 or nx1 > rx1+rw/5 or ny2<ry2-rh/5 or ny1 >ry1+rh/5:
                    continue

                dx = float(rcx - ncx)/float(nw) * ScaleFacetors[0]
                dy = float(rcy - ncy)/float(nh) * ScaleFacetors[1]
                dw = np.log(float(rw)/float(nw)) * ScaleFacetors[2]
                dh = np.log(float(rh)/float(nh)) * ScaleFacetors[3]

                ncropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                crop_box = np.array([nx1, ny1, nx2, ny2])

                overlap_flag = 0
                if nbox > 1:
                    otherboxes = np.array([])
                    for otherbox_idx in xrange(f_boxes.shape[0]):
                        if not box_idx == otherbox_idx:
                            iou = IOU(crop_box, f_boxes[otherbox_idx])
                            #otherboxes = np.append(otherboxes, f_boxes[otherbox_idx])
                            if iou > 0.1:
                                overlap_flag = 1


                if overlap_flag == 1:
                    continue

                Iou = overlapSelf(crop_box, box)
                if np.max(Iou) < THpos:
                    continue


                nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize), interpolation=cv2.INTER_LINEAR)
                box_ = box.reshape(1, -1)

                filename = "/" + str(p_idx) + '_' +  im_path.split('.')[0] + '.jpg'

                save_file = os.path.join(save_dir + filename)
                fw.write(save_dir + filename + ' 1 %.5f %.5f %.5f %.5f\n' % (dx, dy, dw, dh))
                cv2.imwrite(to_dir + save_file, nresized_im)

                p_idx += 1
            box_idx += 1

fw.close()

