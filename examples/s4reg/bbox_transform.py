import numpy as np
import numpy.random as npr
from utils import IOU, overlapSelf, overlapingOtherBox
ScaleFacetors = np.array([10,10,5,5])

def validBox(box, width, height):
    rx1, ry1, rx2, ry2 = box
    if rx1 >= rx2 or ry1 >= ry2:
        return False
    rw = rx2 - rx1
    rh = ry2 - ry1
    rcx = rx1 + rw / 2
    rcy = ry1 + rh / 2

    if max(rw, rh) < 60 or rx1 < 0 or ry1 < 0:
        return False
    if rx1 > rx2 or ry1 > ry2:
        print ":", x1, y1, x2, y2, "-", width, height
    if rx2 > width or ry2 > height or rx1 < 0 or ry1 < 0:
        return False

    return True
# gt_outside: pix allowed to go outside ground truth box
# p_ratio: False - crop a square ; True - crop a reactage
def crop4cls(box, enlarge_bottom, enlargeTop, shift, gt_outside=10, p_ratio=False):
    # TODO: validBox and enlarge params
    # newBox = np.array([])

    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    nw = npr.randint(np.ceil(w * enlarge_bottom), np.ceil(w * enlargeTop))

    if p_ratio == False:
        nh = nw
    else:
        nh = int(h*nw/w)

    delta_x = npr.randint(-int(float(w) * shift), int(float(w) * shift + 1))
    delta_y = npr.randint(-int(float(h) * shift), int(float(h) * shift + 1))

    ncx = cx + delta_x
    ncy = cy + delta_y

    nx1 = ncx - nw / 2
    ny1 = ncy - nh / 2
    nx2 = ncx + nw / 2
    ny2 = ncy + nh / 2

    if nx2 < x2 + gt_outside or nx1 > x1 - gt_outside or ny2 < y2 + gt_outside or ny1 > y1 - gt_outside:
        return np.array([])
    else:
        return np.array([nx1, ny1, nx2, ny2])

def crop4reg(idx, boxes, enlarge_bottom, enlargeTop, shift, gt_outside=10, loop=100):
    # TODO: validBox and enlarge params
    # do not support crop by h/w
    box = boxes[idx]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2

    # maxWH = np.max((w, h))
    # minWH = np.min((w, h))
    meanWH = w + float(h-w)/2.
    i = 0
    while i< loop:
        i+=1
        # enlargeS = npr.randint(np.ceil(maxWH * enlarge_bottom), np.ceil(maxWH * enlargeTop))
        # enlargeS = npr.randint(np.ceil(minWH * enlarge_bottom), np.ceil(minWH * enlargeTop))
        if (enlarge_bottom == 0):
            enlargeS = int(meanWH * enlargeTop)
        else:
            enlargeS = npr.randint(np.ceil(meanWH * enlarge_bottom), np.ceil(meanWH * enlargeTop))

        delta_x = npr.randint(-int( enlargeS * shift), int( enlargeS * shift + 1))
        delta_y = npr.randint(-int( enlargeS * shift), int( enlargeS * shift + 1))

        ncx = cx + delta_x
        ncy = cy + delta_y

        nx1 = ncx - enlargeS / 2
        ny1 = ncy - enlargeS / 2
        nx2 = ncx + enlargeS / 2
        ny2 = ncy + enlargeS / 2

        dx = float(cx - ncx)/float(enlargeS) * ScaleFacetors[0]
        dy = float(cy - ncy)/float(enlargeS) * ScaleFacetors[1]
        dw = np.log(float(w)/float(enlargeS)) * ScaleFacetors[2]
        dh = np.log(float(h)/float(enlargeS)) * ScaleFacetors[3]
        crop_box = np.array([nx1, ny1, nx2, ny2])

        if nx2 < x2 + gt_outside or nx1 > x1 - gt_outside or ny2 < y2 + gt_outside or ny1 > y1 - gt_outside:
            continue
        elif overlapingOtherBox(crop_box, idx, boxes):
            continue
        else:
            return np.array([nx1, ny1, nx2, ny2]), np.array([dx,dy,dw,dh])

    return np.array([]), np.array([])