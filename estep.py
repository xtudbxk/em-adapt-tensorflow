import numpy as np
import random
import time
import copy
from ctypes import POINTER, c_int, c_bool, c_float, CDLL,cdll
import os
import pickle

libweaklabel_path = "libweaklabel.so"
if os.path.exists(libweaklabel_path) is False:
    libweaklabel_path = os.path.join(os.getcwd(),"pythonlib","estep",libweaklabel_path)
    assert os.path.exists(libweaklabel_path),"error, cannot find libweaklabel.so"

estep_lib = cdll.LoadLibrary(libweaklabel_path)
estep_lib.e_step.argtypes=[POINTER(c_float),POINTER(c_int),POINTER(c_int),POINTER(c_int),c_bool,c_int,c_float,c_float,c_float]
#estep_lib.e_step.argtypes=[POINTER(c_float),POINTER(c_int),POINTER(c_int)]

def estep(feature_map,label,suppress_others=True,num_iter=5,margin_others=1e-5,bg_p=0.5,fg_p=0.25,use_c=False):
    #s = time.time()
    if use_c is True:
        feature_map = feature_map.astype(c_float)
        label = label.astype(c_int)
        f = estep_c(feature_map,label,suppress_others,num_iter,margin_others,bg_p,fg_p)
    else:
        f = estep_py(feature_map,label,suppress_others,num_iter,margin_others,bg_p,fg_p)
    #print("duration:%f" % (time.time()-s))
    return f.astype(np.float32)
    
def estep_c(feature_map,label,suppress_others,num_iter,margin_others,bg_p,fg_p):
    estep_lib.e_step(feature_map.ctypes.data_as(POINTER(c_float)),feature_map.ctypes.shape_as(c_int),feature_map.ctypes.strides_as(c_int),label.ctypes.data_as(POINTER(c_int)),suppress_others,num_iter,margin_others,bg_p,fg_p)
    return feature_map

def estep_py(feature_map,label,suppress_others,num_iter,margin_others,bg_p,fg_p):
    #print("before label:%s" % str(label))
    label = label.astype(np.uint8)
    label_ = np.zeros([feature_map.shape[0],feature_map.shape[3]],dtype=np.uint8)
    #print("before label:%s" % str(label_))
    for i in range(feature_map.shape[0]):
        index = np.unique(label[i])
        #print("index:%s" % str(index))
        for one in index:
            if one >= feature_map.shape[3]: continue
            label_[i,one] = 1
    label = label_
    #print("after label:%s" % str(label))
    if suppress_others is True:
        mask = np.repeat(label,feature_map.shape[1]*feature_map.shape[2],axis=0)
        mask = np.reshape(mask,[-1,feature_map.shape[1],feature_map.shape[2],feature_map.shape[3]])
        tmp_map = copy.deepcopy(feature_map)
        tmp_map[mask<1] += np.amax(feature_map)
        min_each_image = np.amin(tmp_map,axis=3)
        min_each_image = np.repeat(min_each_image,feature_map.shape[3],axis=2)
        min_each_image = np.reshape(min_each_image,[-1,feature_map.shape[1],feature_map.shape[2],feature_map.shape[3]])
        mask = np.logical_and(mask<1,feature_map>min_each_image)
        feature_map[mask] = min_each_image[mask] - margin_others
    before_mean_value = np.mean(np.amax(feature_map,axis=3),(1,2))
    order = [k for k in range(feature_map.shape[3])]
    b_th = int(feature_map.shape[1]*feature_map.shape[2]*bg_p)
    f_th = int(feature_map.shape[1]*feature_map.shape[2]*fg_p)
    #print("before_mean_value:%s" % str(before_mean_value))
    #print("after suppress_others:%s" % str(feature_map))
    for i in range(num_iter):
        #print("order:%s" % str(order))
        tmp_ = order[1:]
        random.shuffle(tmp_)
        tmp_.insert(0,0)
        #print("after random order:%s" % str(tmp_))
        for j in tmp_:
            for i in range(feature_map.shape[0]):
                if label[i][j] > 0:
                    diff_value = np.reshape(np.amax(feature_map[i],axis=2)-feature_map[i,:,:,j],[-1])
                    if j == 0:
                        th_value = np.partition(diff_value,b_th)[b_th]
                        #print("diff_value:%s" % str(np.partition(diff_value,b_th)))
                    else:
                        th_value = np.partition(diff_value,f_th)[f_th]
                        #print("diff_value:%s" % str(np.partition(diff_value,f_th)))
                    #print("th_value:%s" % str(th_value))
                    feature_map[i,:,:,j] += th_value
        #print("map:%s" % str(feature_map))
    after_mean_value = np.mean(np.amax(feature_map,axis=3),(1,2))
    feature_map += np.reshape(before_mean_value-after_mean_value,[-1,1,1,1])
    #print("after_mean_value:%s" % str(after_mean_value))
    return feature_map
