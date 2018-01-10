import numpy as np
import random
import time
import copy
from ctypes import POINTER, c_int, c_bool, c_float, CDLL,cdll

bg_p = 0.5
fg_p = 0.25
num_iter = 5
suppress_others = True
margin_others = 1e-5
batch_size = 20
h,w = 100,100
categorys = 10

estep_lib = cdll.LoadLibrary("./libestep.so")
estep_lib.e_step.argtypes=[POINTER(c_float),POINTER(c_int),POINTER(c_int),POINTER(c_int),c_bool,c_int,c_float,c_float,c_float]
#estep_lib.e_step.argtypes=[POINTER(c_float),POINTER(c_int),POINTER(c_int)]
estep_lib.e_step.restypes=np.ctypeslib.ndpointer

def estep(feature_map,label):
    r = estep_lib.e_step(feature_map.ctypes.data_as(POINTER(c_float)),feature_map.ctypes.shape_as(c_int),feature_map.ctypes.strides_as(c_int),label.ctypes.data_as(POINTER(c_int)),suppress_others,num_iter,margin_others,bg_p,fg_p)
    #r = estep_lib.e_step(feature_map.ctypes.data_as(POINTER(c_float)),feature_map.ctypes.shape_as(POINTER(c_int)),feature_map.ctypes.strides_as(POINTER(c_int)))
    #print("after map:%s" % str(feature_map))
    #print("after r:%s" % str(r))
    return feature_map


if __name__ == "__main__":
    f = np.random.randn(batch_size,h,w,categorys)
    l = np.array([[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1],[1,1,0,0,1,0,0,0,1,1],[1,0,1,0,0,0,1,1,1,1]][:batch_size])
    #f = np.array([[[[1,2,3]],[[3,2,1]]],[[[4,5,6]],[[7,8,9]]]],dtype=np.float32) # shape: 2x2x1x3
    #l = np.array([[1,1,0],[1,0,1]]) # shape: 2x3
    #f = np.array([[[[1,2,3]],[[3,2,1]]]],dtype=np.float32) # shape: 2x2x1x3
    #l = np.array([[1,1,1]]) # shape: 2x3
    #print("before:%s" % str(f))
    #print("before:%s" % str(l))
    start_time = time.time()
    #print("start_time:%f" % start_time)
    t = estep(f,l)
    #print("after:%s" % str(t))
    end_time = time.time()
    #print("end_time:%f" % end_time)
    print("duration time:%f" % (end_time-start_time))
