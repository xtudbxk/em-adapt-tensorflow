import os
import re
import sys
import glob
import json
import numpy as np 
import skimage
import skimage.io as imgio
from metrics import metrics_np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

def crf_inference(pred_or_feat,img,crf_config,categorys_num,gt_prob=0.7,softmax=True):
    '''
    feat: the feature map of cnn, shape [h,w,c] or pred, shape [h,w], float32
    img: the origin img, shape [h,w,3], uint8
    crf_config: {"g_sxy":3,"g_compat":3,"bi_sxy":5,"bi_srgb":5,"bi_compat":10,"iterations":5}
    '''
    h,w = img.shape[0:2]
    crf = dcrf.DenseCRF2D(h,w,categorys_num)

    if len(pred_or_feat.shape) == 3 and pred_or_feat.shape[2] != 1:
        pred_or_feat = np.exp(pred_or_feat)
        pred_or_feat /= np.expand_dims(np.sum(pred_or_feat,axis=2),axis=2)
        unary = -np.log(pred_or_feat)
        unary = np.reshape(unary,(-1,categorys_num))
        unary = np.swapaxes(unary,0,1)
        unary = np.copy(unary,order="C")
        crf.setUnaryEnergy(unary)
    else:
        # unary energy
        unary = unary_from_labels(pred_or_feat,categorys_num,gt_prob,zero_unsure=False)
        crf.setUnaryEnergy(unary)

    # pairwise energy
    crf.addPairwiseGaussian( sxy=crf_config["g_sxy"], compat=crf_config["g_compat"] )
    crf.addPairwiseBilateral( sxy=crf_config["bi_sxy"], srgb=crf_config["bi_srgb"], rgbim=img, compat=crf_config["bi_compat"] )
    Q = crf.inference( crf_config["iterations"] )
    r = np.argmax(Q,axis=0).reshape((h,w))
    return r

def crf_grid_search():
    from predict import Predict
    from dataset import dataset_np
    config = {"g_sxy":3,"g_compat":3,"bi_compat":5,"bi_sxy":50,"bi_srgb":10,"iterations":10}
    for bi_compat_ in range(1,11,3):
        config["bi_compat"] = bi_compat_
        for bi_sxy_ in range(10,110,30):
            config["bi_sxy"] = bi_sxy_
            for bi_srgb_ in range(1,22,5):
                config["bi_srgb"] = bi_srgb_
                print("crf config:%s" % str(config))
                #print("start to inference ...")
                _crf_grid_search(config)
                print("\n")

def _crf_grid_search(crf_config):
    '''
    input img is xxx_origin.png, gt is xxx_gt.png, pred is xxx_pred.png, output npy is xxx_output.npy
    '''
    category_num = 21
    main_path = "preds"
    img_filenames_list = glob.glob(os.path.join(main_path,"*_origin.png"))
    img_filenames_list = img_filenames_list[:200]
    #print("img_filenames_list:%s" % str(img_filenames_list))

    m = metrics_np(n_class=category_num)
    for img_filename in img_filenames_list:
        id = os.path.basename(img_filename)[:-11]
        gt_filename = os.path.join(main_path,"%s_gt.png" % id)
        if not os.path.exists(gt_filename): continue
        #pred_filename = os.path.join(main_path,"%s_pred.png" % id)
        #if os.path.exists(pred_filename): continue
        output_filename = os.path.join(main_path,"%s_output.npy" % id)
        if not os.path.exists(output_filename): continue

        img = imgio.imread(img_filename)
        img = skimage.img_as_ubyte(img)
        #pred = imgio.imread(pred_filename)
        #pred = skimage.img_as_ubyte(pred)
        output = np.load(output_filename,encoding="latin1")
        output = output.astype(np.float32)
        crf_pred = crf_inference(output,img,crf_config,category_num)
        gt = imgio.imread(gt_filename)
        gt = skimage.img_as_ubyte(gt)
        m.update(gt,crf_pred)
    print("metrics:%s" % str(m.get("miou")))

def visualize_grid_search_result(filepath=None):
    if filepath is None: filepath = sys.argv[1]
    with open(filepath,"r") as f:
        data = {"miou":[],"bi_compat":[],"bi_sxy":[],"bi_srgb":[]}
        for line in f.readlines():
            line = line.strip("\n")
            if "crf config" in line:
                ret = re.findall("(?<=crf config:).*",line)
                assert len(ret) > 0,"len(ret) < 0 while crf config in it"
                #print("ret:%s" % str(ret))
                ret[0] = ret[0].replace("'",'"')
                config = json.loads(ret[0])
                for category in ["bi_compat","bi_sxy","bi_srgb"]:
                    data[category].append(config[category])
            if "metrics" in line:
                ret = re.findall("(?<=metrics:).*",line)
                assert len(ret) > 0,"len(ret) < 0 while metrics in it"
                miou = float(ret[0])
                data["miou"].append(miou)
        for category in data:
            data[category] = np.array(data[category])
    plt.figure
    subplot_n = len(data["bi_compat"])
    
    for i in range(subplot_n):
        plt.subplot(subplot_n,1,i+1)
        plt.title("bi_compat:%s" % data["bi_compat"][i])
        plt.xlabel("bi_srgb")
        plt.ylabel("miou")

        perm = (data["bi_compat"] == data["bi_compat"][i])
        tmp_data = data["data"][perm]
        tmp_bi_sxy = data["bi_sxy"][perm]
        tmp_bi_srgb = data["bi_srgb"][perm]

        bi_sxys = np.unique(tmp_bi_sxy)
        for bi_sxy in bi_sxys:
            perm = (tmp_bi_sxy == bi_sxy)
            tmp_tmp_data = tmp_data[perm]
            tmp_tmp_bi_srgb = tmp_bi_srgb[perm]

            plt.plot(tmp_tmp_bi_srgb,tmp_tmp_data,label="bi_sxy:%d" % bi_sxy)

        plt.legend()
        
        

def test_crf(): # you need to comment the self.load_saver in __init__
    img = np.ones((4,4,3),dtype=np.uint8)
    f = np.ones((4,4),dtype=np.float32)
    f[:0] = 0
    f[:1] = 1
    f[:3] = 2
    f[:2] = 3
    config = {"g_sxy":10,"g_compat":10,"bi_sxy":10,"bi_srgb":10,"bi_compat":10,"iterations":5}
    p = crf_inference(f,img,config,4)
    print("return: %s" % str(p.crf_inference(f,img)))
        
if __name__ == "__main__":
    #test_crf()
    #crf_grid_search()
    visualize_grid_search_result()
