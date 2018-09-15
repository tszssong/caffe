import numpy as np
import copy
import re
import cv2
import os
import json
import math
import random
from time import gmtime, strftime

def ColorJitterAug(image, brightness=0, contrast=0, saturation=0):
    if brightness > 0:
        alpha = 1.0 + random.uniform(-brightness, brightness)
        image = image * alpha
    if contrast > 0:
        coef = np.array([[[0.299, 0.587, 0.114]]])
        alpha = 1.0 + random.uniform(-contrast, contrast)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = image * alpha
        image = image + gray
    if saturation > 0:
        coef = np.array([[[0.299, 0.587, 0.114]]])
        alpha = 1.0 + random.uniform(-saturation, saturation)
        gray = image * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray = gray * (1.0 - alpha)
        image = image * alpha
        image = image + gray

    image = np.clip(image, 0, 255)

    return image

def HueJitterAug(image, hue=0):
    tyiq = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.274, -0.321],
                              [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621],
                            [1.0, -0.272, -0.647],
                            [1.0, -1.107, 1.705]])

    alpha = random.uniform(-hue, hue)
    u = np.cos(alpha * np.pi)
    w = np.sin(alpha * np.pi)
    bt = np.array([[1.0, 0.0, 0.0],
                    [0.0, u, -w],
                    [0.0, w, u]])
    t = np.dot(np.dot(ityiq, bt), tyiq).T
    image = np.dot(image, np.array(t))

    image = np.clip(image, 0, 255)

    return image

def MotionBlurAug(image, blurvalue=0):
    degree = int(random.uniform(1, blurvalue * 20))
    if degree == 0:
        degree = 1
    angle = random.uniform(-180, 180)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree        
    image = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return image

def GaussianBlurAug(image, blurvalue=0):
    degree = int(random.uniform(1, blurvalue * 10))
    if degree % 2 == 0:
        degree += 1
    sigmaX = random.randint(0, 5)
    sigmaY = random.randint(0, 5)
    image = cv2.GaussianBlur(image, ksize=(degree, degree), sigmaX=sigmaX, sigmaY=sigmaY)
    return image

def GaussianNoiseAug(image):
    row, col, ch = image.shape
    mean = 0
    var = np.random.uniform(0.004, 1)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    cv2.normalize(noisy, noisy, 0, 255, norm_type=cv2.NORM_MINMAX)
    return noisy

def flipAug(image, boxes, arg):
    #print "flip para:",arg
    oriH, oriW, oriC = image.shape
    height, width, channel = image.shape
    fboxes = np.array([])
    im = copy.deepcopy(image)
    
    if arg == 0:                     #horizontal flip
        im = cv2.flip(im,1)
    elif arg == 1:                   #vertical flip
        im = cv2.flip(im,0)
    elif arg == 2:                   #horizontal and vertical flip
        im = cv2.flip(im,-1)
    elif arg == 3:                   #rot90
        im = cv2.transpose(im)
    elif arg == 4:                   #rot90 and flip
        im = cv2.transpose(im)
        imnew = copy.deepcopy(im)
        im = cv2.flip(imnew,1)       #horizontal flip
    for box in boxes:                # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = box
        if arg == 0:                 #horizontal flip
            rx2 = width - x1 - 1
            rx1 = width - x2 - 1
        elif arg == 1:               #horizontal flip
            ry2 = height - y1 - 1
            ry1 = height - y2 - 1
        elif arg == 2:               #horizontal and vertical flip
            rx2 = width - x1 - 1
            rx1 = width - x2 - 1
            ry2 = height - y1 - 1
            ry1 = height - y2 - 1
        elif arg == 3:               #rot90
            rx1 = y1
            ry1 = x1
            rx2 = y2
            ry2 = x2
        elif arg == 4:               #rot90 and transpose
            rx1 = y1
            ry1 = x1
            rx2 = y2
            ry2 = x2
            temp = rx2               #take attention!
            rx2 = oriH - rx1 - 1
            rx1 = oriH - temp - 1
        fboxes = np.append( fboxes, np.array([rx1, ry1, rx2, ry2]) )
    return im, fboxes


def rotAug(image, boxes, rot_d):
    im_height, im_width, channel = image.shape
    im = copy.deepcopy(image)
    M = cv2.getRotationMatrix2D((im_width / 2, im_height / 2), rot_d, 1)
    im = cv2.warpAffine(im, M, (im_width, im_height), None, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
    rotate_boxes = np.empty((0, 4), dtype=np.float32)
    for j in xrange(len(boxes)):
        [x1, y1, x2, y2] = [int(x) for x in boxes[j]]
        new_pt1 = np.dot(M, np.array([x1, y1, 1]).transpose()).astype(np.int32).transpose()
        new_pt2 = np.dot(M, np.array([x2, y2, 1]).transpose()).astype(np.int32).transpose()
        new_pt3 = np.dot(M, np.array([x1, y2, 1]).transpose()).astype(np.int32).transpose()
        new_pt4 = np.dot(M, np.array([x2, y1, 1]).transpose()).astype(np.int32).transpose()
        rect_pts = np.array([[new_pt1, new_pt2, new_pt3, new_pt4]])
        x, y, w, h = cv2.boundingRect(rect_pts)
        if x<0 or y<0 or x+w>im_width or y+h >im_height:
            continue
        # offset_x = 0
        # offset_y = 0
        # if x < 0:
        #     offset_x = -x
        #     x = 0
        # if y < 0:
        #     offset_y = -y
        #     y = 0
        # if x + w - offset_x < 0 or y + h - offset_y < 0:
        #     rotate_boxes = []
        #     im = image
        #     break
        # if x + w - offset_x > im_width:
        #     w = im_width - 1 - x - offset_x
        # if y + h - offset_y > im_height:
        #     h = im_height - 1 - y - offset_y
        # rotate_boxes = np.vstack((rotate_boxes, np.array([x, y, x + w, y + h])))
        rotate_boxes = np.append( rotate_boxes, np.array([x, y, x + w, y + h]) )
    return im, rotate_boxes

def ImageAug(image, brighteness=0, contrast=0, saturation=0, hue=0, togray=0):
    image = ColorJitterAug(image, brighteness, contrast, saturation)
    if hue > 0:
        image = HueJitterAug(image, hue)

    blur_flag = random.randint(0, 10)
    if 7 <= blur_flag <= 8:
        image = MotionBlurAug(image, random.random())
    elif 9 <= blur_flag <= 10:
        image = GaussianBlurAug(image, random.random())

    noise_flag = random.randint(0, 10)
    if noise_flag >= 7:
        image = GaussianNoiseAug(image)

    if togray > 0:
        image = image.astype(np.uint8)
        if random.random() < togray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.astype(np.uint8)
