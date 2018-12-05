import sys
import numpy as np
import cv2
import copy
import os
import re
import numpy.random as npr

data_dir_path = wk_dir + "trainData/"
model_dir_path = wk_dir + "ResNet50Anno/"
if __name__=='__main__':
    pattern = re.compile(r'^[^\.].+\.jpg$')
    numPic = 0
    for dirpath, dirnames, filenames in os.walk(data_dir_path):
        for img_filename in filenames:
            match = pattern.match(img_filename)
            if not match:
                continue
            pt_filename = img_filename.replace('.jpg', '.txt')
            model_ptfile_path = os.path.join(model_dir_path, pt_filename)
            human_ptfile_path = os.path.join(dirpath, pt_filename)
            m_lbl = np.loadtxt(model_ptfile_path, dtype=float)
            h_lbl = np.loadtxt(human_ptfile_path, dtype=float)
            delta_lbl = h_lbl - m_lbl
            label_w = m_lbl[2]-m_lbl[0]
            label_h = m_lbl[3]-m_lbl[1]
            normal_lbl = np.array([delta_lbl[0]/label_w, delta_lbl[1]/label_h, delta_lbl[2]/label_w, delta_lbl[3]/label_h])
            numPic += 1
            # print '\n',pt_filename
            # print h_lbl
            # print m_lbl
            # print delta_lbl
            # print delta_lbl[0]/label_w, delta_lbl[1]/label_h, delta_lbl[2]/label_w, delta_lbl[3]/label_h
            # print normal_lbl
            filedflag = False
            for i in xrange(4):
                if abs(delta_lbl[i]) > 2 or abs(normal_lbl[i]>0.02):
                    # print "fixed label:", i, m_lbl[i], h_lbl[i]
                    m_lbl[i] = h_lbl[i]
                    filedflag =  True
            if filedflag:
                fw = open(model_ptfile_path, 'w')
                fw.write('%.5f %.5f %.5f %.5f\n'%(m_lbl[0], m_lbl[1], m_lbl[2], m_lbl[3]))
                fw.close()
            # im_small = cv2.imread(os.path.join(dirpath, img_filename))
            # show_x1 = int(label[0])
            # show_y1 = int(label[1])
            # show_x2 = int(label[2])
            # show_y2 = int(label[3])
            # print show_x1, show_y1, show_x2, show_y2
            # cv2.rectangle(im_small,(show_x1, show_y1),(show_x2,show_y2), showColor, 2 )
            # hx1, hy1, hx2, hy2 = h_lbl.astype(int)
            # cv2.rectangle(im_small, (hx1, hy1), (hx2, hy2), (0, 255, 0), 1)
            # mx1, my1, mx2, my2 = m_lbl.astype(int)
            # cv2.rectangle(im_small, (mx1, my1), (mx2, my2), (0, 255, 255), 1)
            # cv2.imshow("img", im_small)
            # cv2.waitKey()
    print "the end! %d pic in dir"%numPic