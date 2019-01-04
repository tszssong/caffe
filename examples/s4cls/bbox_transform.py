import numpy as np
import numpy.random as npr


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
def crop4cls(box, enlarge_bottom, enlargeTop, shift, gt_outside=10, p_ratio=False, loop = 100):
    # TODO: validBox and enlarge params
    # newBox = np.array([])

    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    i = 0
    while i< loop:
        i+=1
        nw = npr.randint(np.ceil(w * enlarge_bottom), np.ceil(w * enlargeTop))

        if p_ratio == False:
            nh = nw
        else:
            nh = int(h*nw/w)

        # delta_x = npr.randint(-int(float(w) * shift), int(float(w) * shift + 1))
        # delta_y = npr.randint(-int(float(h) * shift), int(float(h) * shift + 1))
        delta_x = npr.randint(-int(nw*shift), int(nw*shift))
        delta_y = npr.randint(-int(nh*shift), int(nh*shift))

        ncx = cx + delta_x
        ncy = cy + delta_y

        nx1 = ncx - nw / 2
        ny1 = ncy - nh / 2
        nx2 = ncx + nw / 2
        ny2 = ncy + nh / 2

        if nx2 < x2 + gt_outside or nx1 > x1 - gt_outside or ny2 < y2 + gt_outside or ny1 > y1 - gt_outside:
            continue
        else:
            return np.array([nx1, ny1, nx2, ny2])
    return np.array([])