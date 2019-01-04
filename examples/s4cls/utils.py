import numpy as np


def IOU(Reframe,GTframe):

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio
def IOU2(Reframe,GTframe):

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area2 - Area +  1)
    return ratio

def IOU_multi(box, boxes):
    maxIou = 0.0
    for box_idx in xrange(boxes.shape[0]):
        compBox = boxes[box_idx]
        iou = IOU2(box, compBox)
        if iou > maxIou:
            maxIou = iou
    # print maxIou
    return maxIou

def containBox(box, boxes):
    contain = False
    for box_idx in xrange(boxes.shape[0]):

        compBox = boxes[box_idx]
        compW = compBox[2] - compBox[0]
        compH = compBox[3] - compBox[1]
        if( box[0]<compBox[0]-compW/10 and box[1]<compBox[1]-compH/10 and box[2]>compBox[2]+compW/10 and box[3]< compBox[3]+compH/10):
            contain = True
    return contain

def overlapSelf(Reframe,GTframe):
    """Compute overlap between detect box and gt boxes

        """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        ratio = Area * 1. / Area1
    return ratio

def overlapingOtherBox(crop_box, box_idx, f_boxes):

    overlap_flag = 0
    for otherbox_idx in xrange(f_boxes.shape[0]):
        if not box_idx == otherbox_idx:
            iou = IOU(crop_box, f_boxes[otherbox_idx])
            if iou > 0.01:
                overlap_flag = 1
    if overlap_flag == 1:
        return True

