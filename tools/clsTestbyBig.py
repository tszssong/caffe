import os, sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
os.environ['GLOG_minloglevel'] = '3'
import numpy as np
import numpy.random as npr
#import scipy.io as sio
import caffe
import cv2
import time
gt_file = "/Users/momo/wkspace/caffe_space/caffe/data/pic/627928skip_resized180.txt"
img_dir = "/Users/momo/wkspace/caffe_space/caffe/data/pic/627928skip_resized180-img/"
xml_dir = "/Users/momo/wkspace/caffe_space/caffe/data/pic/627928skip_resized180-xml/"

prototxt   = "examples/hand_cls/mouth64/test8cls.prototxt"
caffemodel = "models/8clsmouth64/8cls_1w+2w_185k.caffemodel"
NumTest = 20000
if __name__ == '__main__':
    caffe.set_mode_cpu()
    inputSize = 64
    mean = np.array([104, 117, 123])
    classify_net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    fid = open(gt_file,"r")
    subdirlists = ['bg', 'heart', 'yearh', 'one', 'baoquan', 'five', 'bainian', 'zan', 'fingerheart', 'other']
    tp_dict = {}
    gt_dict = {}
    re_dict = {}
    for gname in subdirlists:
        tp_dict[gname] = 0
        gt_dict[gname] = 0
        re_dict[gname] = 0
    TP=0
    err = 0
    lines = fid.readlines()
    fid.close()
    cur_=0
    sum_=len(lines)
    regloss = np.array([])
    probs = np.array([])
    cls = 0
    totalTime = 0
    for line in lines:
        cur_+=1
        if not line or cur_ == NumTest:
            break;
        words = line.split()
#        image_file_name = "data/clsData/" + words[0]
        image_file_name = "data/48Test/" + words[0]

        if cur_%500 == 0:
            print cur_,
            sys.stdout.flush()
        im = cv2.imread(image_file_name)
        h,w,ch = im.shape
        if h!=inputSize or w!=inputSize:
            im = cv2.resize(im,(int(inputSize),int(inputSize)))
        copy = im

        im = im.astype(np.int)
        im -= mean
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        label    = int(words[1])
        if label > 13:
            continue
        gt_dict[subdirlists[label]] += 1

        classify_net.blobs['data'].reshape(1,3,inputSize,inputSize)
        classify_net.blobs['data'].data[...]=im
        out_ = classify_net.forward()
        prob = out_['prob'][0]

        cls_prob = np.max(prob)
        cls = np.where(prob==np.max(prob))[0][0]
        re_dict[subdirlists[cls]] += 1
        if not cls == label:
            err += 1
            savename =  words[0].split('/')[-1].split('.')[0] + '_' + str(cls) + '.jpg'
            cv2.imwrite("/Users/momo/wkspace/caffe_space/caffe/data/48Test/error/" + savename, copy)
        else:
            tp_dict[subdirlists[cls]]+=1
    print "\nerr, sum:",err, sum_
    reTotal = 0
    gtTotal = 0
    tpTotal = 0
    for gname in tp_dict:
        print "%12s"%(gname)," Recall:%.2f"%( float(tp_dict[gname])/float(gt_dict[gname]+1) ), \
              " Precision:%.2f"%( float(tp_dict[gname])/float(re_dict[gname]+1) ), \
              " re:%4d"%(re_dict[gname]), " tp:%4d"%(tp_dict[gname]), "gt:%4d"%(gt_dict[gname])
        
        reTotal += re_dict[gname]
        gtTotal += gt_dict[gname]
        tpTotal += tp_dict[gname]
    print "total recall:%.2f"%(float(tpTotal)/float(gtTotal)), "total precision:%.2f"%(float(tpTotal)/float(reTotal))
    print prototxt
    print caffemodel

