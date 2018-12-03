import sys
import numpy as np
import cv2
import copy
import os
import numpy.random as npr
from bbox_transform import validBox, crop4reg_small
from image_process import crop_image
paddingMode = 'black'
cropSize = 300
ScaleB = 3.0
# im_dir = "/Users/momo/wkspace/gesture/data/ori/T-5-five_21081025pink-img/"
im_dir = "/Volumes/song/gestureDatabyName/2-yearh-img/"
anno_file = "gt/total/"+sys.argv[1]
#anno_file = "gt/T_5_five-pink.txt"
# im_dir = "/Volumes/song/handg_pink/T-13-fist_20181026pink-img/"
# anno_file = "/Users/momo/wkspace/caffe_space/detection/caffe/examples/s4reg/gt/T_13_fist-pink.txt"
to_dir = "../../data/trainData/"
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
    #print im_path
    nbox = int(annotation[1])
    if nbox>2:
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
    height, width, channel = image.shape
    for box_idx in xrange(f_boxes.shape[0]):
        box = f_boxes[box_idx]
        crop_box, reg_coord = crop4reg_small(box_idx, f_boxes,  ScaleB)
        if not crop_box.size == 4:
            continue
        ncropped_im = crop_image(image, crop_box, paddingMode)
        nresized_im = cv2.resize(ncropped_im, (cropSize, cropSize))
        ratioW = float(cropSize)/(crop_box[2]-crop_box[0])
        ratioH = cropSize/float(crop_box[3]-crop_box[1])
        box_ = box.reshape(1, -1)
        filename = "/" + str(croped_pic_idx) + '_' +  im_path.split('.')[0]
        save_file = os.path.join(save_dir + filename)

        cv2.imwrite(to_dir + save_file + '.jpg', nresized_im)
        annofw = open(to_dir + save_file +'.txt', 'w')
        annofw.write('%.5f %.5f %.5f %.5f\n' % (reg_coord[0]*ratioW, reg_coord[1]*ratioH, reg_coord[2]*ratioW, reg_coord[3]*ratioH) )
        annofw.close()
        croped_pic_idx += 1
        box_idx += 1
print croped_pic_idx, "images croped"


