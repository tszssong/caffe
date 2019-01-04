import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from bbox_utils import IOU, overlapingOtherBox, validBox, crop4reg_small

cropSize = 320
ScaleB = 3.0
# im_dir = "/Volumes/song/handgesture5_48G/Tight_ali2_five_test-img/"
# anno_file = "/Volumes/song/handgesture5_48G/Tight_ali2_five_test-xml.txt"
# im_dir = "/Volumes/song/handgesture5_48G/Tight_ali2_five_train-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_5_ali2.txt"
# im_dir = "/Volumes/song/gestureTight4Reg/Tight5-notali2-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_5_hebing.txt"
# anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_5_notali2_ali2grab.txt"
# im_dir = "/Volumes/song/gestureDatabyName/2-yearh-img/"
# im_dir = "/Volumes/song/gestureDatabyName/3-one-img/"
# im_dir = "/Volumes/song/gestureDatabyName/7-zan-img/"
# im_dir = "/Volumes/song/gestureDatabyName/8-fingerheart-img/"
# im_dir = "/Volumes/song/gestureDatabyName/12-big_v-img/"
# im_dir = "/Volumes/song/gestureDatabyName/13-fist-img/"
# im_dir = "/Volumes/song/handg_neg_test32G/20181018wsRegTest/wsRegTest-img/"
# anno_file = "gt_sep/"+sys.argv[1]
im_dir = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/gzAndroidTest/JPEGImages/"
anno_file = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/gzAndroidTest/gt_test.txt"
im_dir = "/Users/momo/Downloads/TightOther-img/"
anno_file = "/Users/momo/Downloads/T-Other-badcase3.txt"
# im_dir = "/Volumes/song/handg_pink/T-13-fist_20181026pink-img/"
to_dir = "/Users/momo/wkspace/gesture/handreg/data/"
# to_dir = "/Users/momo/wkspace/gesture/test_regGesture/data/testData/"
annofileName = anno_file.split('.')[0].split('/')[-1]
save_name = annofileName +'_' + str(cropSize)+ 'S'+ str(int(ScaleB * 10))
save_dir = save_name

if not os.path.exists(to_dir+save_dir):
    os.mkdir(to_dir+save_dir)
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print anno_file.split('/')[-1]
print "%d pics in total" % num
croped_pic_idx = 0  # positive
ori_pic_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    nbox = int(annotation[1])
    if nbox>2:     # skip some annocation error
        continue

    bbox = map(float, annotation[3:])
    f_boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

    image = cv2.imread(os.path.join(im_dir, im_path))
    try:
        image.shape
    except:
        print im_path, "not exist!"
        continue

    ori_pic_idx += 1
    if ori_pic_idx % 100 == 0:
        print "%s images done, pos: %s"%(ori_pic_idx, croped_pic_idx)

    for box_idx in xrange(f_boxes.shape[0]):
        box = f_boxes[box_idx]
        if nbox > 1:
            overlapflag, overlaplists = overlapingOtherBox(box, box_idx, f_boxes, 0.03)
            if overlapflag:
                continue

        crop_box, reg_coord = crop4reg_small(box_idx, f_boxes,  ScaleB)

        if not crop_box.size == 4:
            continue

        cbx1,cby1,cbx2,cby2 = crop_box
        M_tmp = cv2.getRotationMatrix2D( (cbx1, cby1), 0, 1 )
        M_tmp[:,2] -= (cbx1, cby1)
        if nbox > 1:
            overlapflag, overlaplists = overlapingOtherBox(crop_box, box_idx, f_boxes, 0.01)
            if overlapflag:
                copyImg = image.copy()
                for black_idx in overlaplists:
                    bx1, by1, bx2, by2 = f_boxes[black_idx].astype(np.int32)
                    copyImg[by1:by2, bx1:bx2] = np.zeros([by2-by1, bx2-bx1, 3])
                ncropped_im = cv2.warpAffine(image, M_tmp, (int(cbx2-cbx1), int(cby2-cby1)))
        else:
            ncropped_im = cv2.warpAffine(image, M_tmp, (int(cbx2-cbx1), int(cby2-cby1)))

        if ncropped_im.shape[0] < cropSize / ScaleB or ncropped_im.shape[1] < cropSize / ScaleB:
            print im_path, "cropped shape:", ncropped_im.shape
            continue

        nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize))
        ratioW = float(cropSize)/(crop_box[2]-crop_box[0])
        ratioH = cropSize/float(crop_box[3]-crop_box[1])

        filename = "/" + im_path.split('.')[0] + 'c'  + str(croped_pic_idx)
        save_file = os.path.join(save_dir + filename)
        cv2.imwrite(to_dir + save_file + '.jpg', nresized_im)
        annofw = open(to_dir + save_file +'.txt', 'w')
        annofw.write('%.5f %.5f %.5f %.5f\n' % (reg_coord[0]*ratioW, reg_coord[1]*ratioH, reg_coord[2]*ratioW, reg_coord[3]*ratioH) )
        annofw.close()
        croped_pic_idx += 1
        box_idx += 1
print croped_pic_idx, "images croped"


