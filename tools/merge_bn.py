import os, sys
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/build/python')
sys.path.append('/Users/momo/wkspace/caffe_space/detection/caffe/python')
import numpy as np
import sys,os
import caffe  

train_proto = 'examples/hand_cls/mouth48/bnTest.prototxt'
train_model = 'models/mouth48_1012/adbg_drop_iter_850000.caffemodel'  #should be your snapshot caffemodel

deploy_proto = 'examples/hand_cls/mouth48/mergebn14cls.prototxt'
save_model = 'examples/hand_cls/mouth48/bn65w.caffemodel'

def merge_bn(net, nob):
    '''
    merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
    var = var + scaleFacotr
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''
    for key in net.params.iterkeys():
#        print key
        if type(net.params[key]) is caffe._caffe.BlobVec:
#            if key.endswith("_bn") or key.endswith("_scale"):
            if "/bn" in key or "_scale" in key:
                continue
            else:
                conv = net.params[key]
                if not( net.params.has_key(key + "/bn") or net.params.has_key(key + "_bn") ):
                    for i, w in enumerate(conv):
                        nob.params[key][i].data[...] = w.data
                else:
                    print key
                    bn = net.params[key + "/bn"]
                    scale = net.params[key + "_bn_scale"]
                    if key == "conv5":
                        bn = net.params[key + "_bn"]
                    
                    wt = conv[0].data
                    channels = wt.shape[0]
                    bias = np.zeros(wt.shape[0])
                    if len(conv) > 1:
                        bias = conv[1].data
                    mean = bn[0].data
                    var = bn[1].data
                    scalef = bn[2].data

                    scales = scale[0].data
                    shift = scale[1].data

                    if scalef != 0:
                        scalef = 1. / scalef
                    mean = mean * scalef
                    var = var * scalef
                    rstd = 1. / np.sqrt(var + 1e-5)
                    rstd1 = rstd.reshape((channels,1,1,1))
                    scales1 = scales.reshape((channels,1,1,1))
                    wt = wt * rstd1 * scales1
                    bias = (bias - mean) * rstd * scales + shift
                    
                    nob.params[key][0].data[...] = wt
                    nob.params[key][1].data[...] = bias
  

net = caffe.Net(train_proto, train_model, caffe.TRAIN)  
net_deploy = caffe.Net(deploy_proto, caffe.TEST)  

merge_bn(net, net_deploy)
net_deploy.save(save_model)

