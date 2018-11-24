# -*- coding: utf-8 -*-
import sys
import os
import re
sys.path.append("/usr/local/Cellar/opencv/3.4.1_2/lib/python2.7/site-packages")
import cv2
import random
import xml.etree.cElementTree as ET
from xml.etree.cElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import numpy as np

# import matplotlib.pyplot as plt
fileName = '/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/small.txt'
# xml_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/" + sys.argv[1] +'/'
# pic_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/"+ sys.argv[1].split('_')[0]+"/"
xml_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations/"
pic_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/"
newPic_path = xml_path + '../new' + sys.argv[1].split('_')[0] +'/'
newxml_path = xml_path + '../new' + sys.argv[1] + '/'
if not os.path.exists(newPic_path):
    os.mkdir(newPic_path)
if not os.path.exists(newxml_path):
    os.mkdir(newxml_path)
distriList = [2,3,4,5,6,8,10]
distriDict = { 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0 }
distribute
scaleB = 10
scaleG = 50

# for dirpath,dirnames,filenames in os.walk(xml_path):
for line in open(fileName, 'r'):
    xml_name = line.strip('\n') + ".xml"
    tree = ET.parse(xml_path+"/"+xml_name)  # 打开xml文档
    root = tree.getroot()  # 获得root节点
    filename = root.find('filename').text
    im = cv2.imread(pic_path+filename)

    size = root.find('size')
    oriW = int(size.find('width').text)
    oriH = int(size.find('height').text)
    if not (oriH - im.shape[0]<2 and oriW - im.shape[1] <2):
        print filename, "size error:",im.shape, oriH, oriW
    num = 0
    objNameList = []
    for object in root.findall('object'):
        num += 1
        name = object.find('name').text  # 子节点下节点name的值
        # print name
        if not name in objNameList:
            objNameList.append(name)

    if num == 0 or len(objNameList)>1:
        continue                         #跳过一张图中多种手
    else:
        m.write(filename + ' ' + str(num) + ' ' + objNameList[0] +' ')
    n_gesture_per_img = 0
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        n_gesture_per_img += 1

        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        if xmin > xmax:
            tmp = xmin
            xmin = xmax
            xmax = tmp
            # print "xml x anno err: ", filename, xmin, xmax
        if ymin > ymax:
            tmp = ymin
            ymin = ymax
            ymax = tmp
            # print "xml y anno err: ", filename, ymin, ymax
        # print xmin, ymin, xmax, ymax, oriH, oriW, np.min([oriW, oriH])
        box_min_side = np.min([xmax-xmin, ymax - ymin])
        im_min_side = np.min([oriW, oriH])
        if xmax-xmin < 4 or ymax-ymin<4:
            print "xml size err: ", filename, xmax, xmin, ymax, ymin
            continue

        scale = int(im_min_side/box_min_side)
        if scale>6:
            newfilename = "crop_" + str(n_gesture_per_img) + '_' + filename.split('.')[0]
            nx1 = np.max([0, xmin-random.randint(scaleB, scaleG)])
            left = xmin - nx1
            ny1 = np.max([0, ymin-random.randint(scaleB, scaleG)])
            up = ymin - ny1
            nx2 = np.min([oriW, xmax+random.randint(scaleB, scaleG)])
            right = nx2 - xmax
            ny2 = np.min([oriH, ymax+random.randint(scaleB, scaleG)])
            down = ny2 - ymax
            print right, down, nx2, ny2, oriH, oriW
            roiImg = im[ny1:ny2, nx1:nx2, :]
            cv2.imshow("roi", roiImg)
            cv2.waitKey(1)
            demoxml=open(os.path.join(newxml_path, newfilename +'.xml'), 'wb+')
            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'GTSDB'

            node_filename = SubElement(node_root, 'filename')
            node_filename.text = newfilename + '.jpg'

            node_size = SubElement(node_root, 'size')
            node_width = SubElement(node_size, 'width')
            node_width.text = str(nx2-nx1)

            node_height = SubElement(node_size, 'height')
            node_height.text = str(ny2-ny1)

            node_depth = SubElement(node_size, 'depth')
            node_depth.text = '3'

            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = 'hand'
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_difficult = SubElement(node_object, 'flag')
            node_difficult.text = '1'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(left)
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(up)
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(left + xmax - xmin)
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(up + ymax - ymin)

            xml = tostring(node_root)  # 格式化显示，该换行的换行
            dom = parseString(xml)

            demoxml.write(xml)

            demoxml.close()
            cv2.imwrite(newPic_path + newfilename + '.jpg', roiImg)


        if scale<2:
            distriDict[2] += 1
        elif scale>10:
            distriDict[10] += 1
        else:
            distriDict[scale] += 1

    # print objNameList

    m.write('\n')
print distriDict

m.close()

