import caffe
import pickle
import numpy as np
import time


def convert(prototxt_path,model_path,saved_path,use_pickle=False):
    layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8_weak']
    net = caffe.Net(prototxt_path,model_path)
    
    params = {}
    for layer in layers:
        params[layer] = net.params[layer][0].data
    
    print("first:%f" % time.clock())
    if use_pickle is True:
        with open(saved_path,"wb") as f:
            pickle.dump(params,f)
    else:
        tmp = np.asarray(params)
        np.save(saved_path,tmp)
    print("second:%f" % time.clock())

if __name__ == "__main__":
    prototxt_path = "weak2/config/adapt/test_val.prototxt"
    model_path = "weak2/model/adapt/init.caffemodel"
    #saved_path = "init.pickle"
    #convert(prototxt_path,model_path,saved_path,use_pickle=True)
    saved_path = "init.npy"
    convert(prototxt_path,model_path,saved_path)


