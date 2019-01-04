import numpy as np
import copy
import cv2
import os
#param: black-enlarge ori pic with black border, else with replicate
def crop_image(img, crop_box, param='black'):
    height, width, channel = img.shape
    right_x = 0
    left_x = 0
    top_y = 0
    down_y = 0
    nx1, ny1, nx2, ny2 = crop_box
    if nx2 > width or ny2 > height or nx1 < 0 or ny1 < 0:
        if nx2 > width:
            right_x = nx2 - width
        if ny2 > height:
            down_y = ny2 - height
        if nx1 < 0:
            left_x = 0 - nx1
        if ny1 < 0:
            top_y = 0 - ny1

        if param == 'black':
            black = [0, 0, 0]
            constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_CONSTANT,
                                      value=black);
        else:
            constant = cv2.copyMakeBorder(img, int(top_y), int(down_y), int(left_x), int(right_x), cv2.BORDER_REPLICATE);
    else:
        constant = copy.deepcopy(img)
    # constant
    return constant[int(ny1 + top_y):int(ny2 + top_y), int(nx1 + left_x):int(nx2 + left_x), :]

def fliterDim(cropped_im):
    b = np.mean(cropped_im[:, :, 0])
    g = np.mean(cropped_im[:, :, 1])
    r = np.mean(cropped_im[:, :, 2])
    if ((b - g) < 10 and abs(b - r) < 10):  # blur img
        return False
    grey = 0.11 * b + 0.59 * r + 0.3 * g
    if (grey < 70):  # dark img
        return False
    return True
