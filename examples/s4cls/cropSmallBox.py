# -*- coding: utf-8 -*-
import sys
import os
import re
sys.path.append("/usr/local/Cellar/opencv/3.4.1_2/lib/python2.7/site-packages")
import cv2
import random
import xml.etree.cElementTree as ET
import numpy as np
gtName = sys.argv[1]+'.txt'
xml_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/" + sys.argv[1] +'/'
pic_path = "/Users/momo/wkspace/caffe_space/detection/py-faster-rcnn/data/VOCdevkit2007/VOC2007/"+ sys.argv[1].split('_')[0]+"/"
newPic_path = xml_path + '../new' + sys.argv[1].split('_')[0] +'/'
newxml_path = xml_path + '../new' + sys.argv[1] + '/'
if not os.path.exists(newPic_path):
    os.mkdir(newPic_path)
if not os.path.exists(newxml_path):
    os.mkdir(newxml_path)
distriList = [2,3,4,5,6,8,10]
distriDict = { 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0 }
m = open(gtName, "w")

for dirpath,dirnames,filenames in os.walk(xml_path):
    for xml_name in filenames:
        if not '.xml' in xml_name:
            continue
        # print f

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

        for object in root.findall('object'):  # 找到root节点下的所有object节点

            bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            if xmin > xmax:
                tmp = xmin
                xmin = xmax
                xmax = tmp
                print "xml x anno err: ", filename, xmin, xmax
            if ymin > ymax:
                tmp = ymin
                ymin = ymax
                ymax = tmp
                print "xml y anno err: ", filename, ymin, ymax

            # print xmin, ymin, xmax, ymax, oriH, oriW, np.min([oriW, oriH])
            box_min_side = np.min([xmax-xmin, ymax - ymin])
            im_min_side = np.min([oriW, oriH])
            if xmax-xmin < 4 or ymax-ymin<4:
                print filename, xmax, xmin, ymax, ymin
                continue


            scale = int(im_min_side/box_min_side)
            if scale>6:
                cv2.imwrite(newPic_path + filename, im)
                tree.write(os.path.join(newxml_path, xml_name), encoding='utf-8')

            # print "scale: ", scale, im_min_side/box_min_side

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

