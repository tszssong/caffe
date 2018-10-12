import numpy as np
from utils import IOU, overlapSelf, IOU_multi, containBox
import copy
import numpy.random as npr
import cv2
# iouH should biger than 0.1 , or arms can not be cropped
def cropNeg(image, boxes,cropSize=128, arg=0, iouTH = 0.1):
    oriH, oriW, oriC = image.shape
    height, width, channel = image.shape
    fboxes = np.array([])
    img = copy.deepcopy(image)
    box = boxes[npr.randint(0,boxes.shape[0])]
    # print box, boxes

    if arg == 0:                     #down
        size = npr.randint(40, min(width, height) / 2)
        gtx1, gty1, gtx2, gty2 = box
        gtheight = int(gty2)-int(gty1)
        nx = int(gtx1)
        ny = int(gty2 - 0.2*gtheight)
        if ny+size>height:
            return None,0
        crop_box = np.array([nx, ny, nx + size, ny + size])

        contain_flag = containBox(crop_box, boxes)
        if contain_flag == True:
            return None,0

        iou = IOU_multi(crop_box, boxes)
        if iou < iouTH:
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
            #resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_LINEAR)
        else:
            return None, 0
    elif arg == 1:                   #right
        size = npr.randint(40, min(width, height) / 2)
        gtx1, gty1, gtx2, gty2 = box
        gtwidth = int(gtx2)-int(gtx1)
        nx = int(gtx2)+ int(0.2*gtwidth)
        ny = int(gty1)
        if nx+size>width:
            return None, 0
        crop_box = np.array([nx, ny, nx + size, ny + size])
        if containBox(crop_box, boxes) == True:
            return None,0

        iou = IOU_multi(crop_box, boxes)
        if iou < iouTH:
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
            #resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_LINEAR)
        else:
            return None, 0

    elif arg == 2:                   #left
        size = npr.randint(40, min(width, height) / 2)
        gtx1, gty1, gtx2, gty2 = box
        gtwidth = int(gtx2)-int(gtx1)
        nx = int(gtx2)+ int(0.2*gtwidth)
        ny = int(gty1)
        if nx+size>width:
            return None, 0
        crop_box = np.array([nx, ny, nx + size, ny + size])
        if containBox(crop_box, boxes) == True:
            return None,0

        iou = IOU_multi(crop_box, boxes)
        if iou < iouTH:
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
            # resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_LINEAR)

        else:
            return None, 0
    else:                   #normal
        size = npr.randint(40, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])
        iou = IOU_multi(crop_box, boxes)
        if containBox(crop_box, boxes) == True:
            return None,0

        if iou < iouTH:
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST)
            #resized_im = cv2.resize(cropped_im, (cropSize, cropSize), interpolation=cv2.INTER_LINEAR)
        else:
            return None, 0
    return resized_im, 1

