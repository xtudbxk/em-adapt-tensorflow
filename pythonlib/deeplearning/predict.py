import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
from scipy import ndimage
import skimage.io as imgio
from dataset import dataset_np
from dataset import dataset_tf
from metrics import metrics_np
import skimage.transform as imgtf
from crf import crf_inference

class Predict():
    def __init__(self,config):
        self.config = config
        self.crf_config = config.get("crf",None)
        self.category_num = self.config.get("category_num",21)
        self.input_size = self.config.get("input_size",(240,240)) # (w,h)
        self.h,self.w = self.input_size
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
        self.sess = tf.Session(config=gpu_options)
        self.net = {}
        self.data = self.config.get("data",None)
        self.load_saver(self.config.get("saver_path","saver/saver"))

    def load_saver(self,saver_filename):
        sess = self.sess
        saver = tf.train.import_meta_graph("%s.meta" % saver_filename)
        self.net["input"] = tf.get_collection("input")[0]
        self.net["prob"] = tf.get_collection("drop_probe")[0]
        self.net["output"] = tf.get_collection("output")[0]
        self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"],(self.h,self.w))
        #self.net["rescale_output"] = tf.get_collection("rescale_output")[0]
        self.net["label"] = tf.placeholder(tf.uint8,[None,self.h,self.w])
        self.net["pred"] = tf.get_collection("pred")[0]
        #self.net["pred"] = tf.argmax(self.net["rescale_output"],axis=3)
        gt,output = self.remove_ignore_label(self.net["label"],self.net["pred"])
        self.mIoU, self.update_op = tf.metrics.mean_iou(gt,output,num_classes=self.category_num)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess,saver_filename)

    def set_crf(self,crf_config):
        self.crf = crf_config

    def remove_ignore_label(self,gt,pred): 
        gt = tf.reshape(gt,shape=[-1])
        pred = tf.reshape(pred, shape=[-1])
        indices = tf.squeeze(tf.where(tf.less(gt,self.category_num)),axis=1)
        gt = tf.gather(gt,indices)
        pred = tf.gather(pred,indices)
        return gt,pred

    def predict_(self,input,single_batch=True,is_pred=False,**kwargs):
        if single_batch is True:
            input = np.reshape(input,[1,input.shape[0],input.shape[1],3])
        params = {self.net["input"]:input,self.net["prob"]:1.0}
        for key in kwargs:
            params[self.net[key]] = kwargs[key]
        if is_pred is True:
            output = self.sess.run(self.net["pred"],feed_dict=params)
        else:
            output = self.sess.run(self.net["rescale_output"],feed_dict=params)
        if single_batch is True:
            output = np.squeeze(output,axis=0)

        return output # note: after softmax, before argmax

    def predict(self,img,single_batch=True,**kwargs):
        output = self.predict_(img,single_batch=single_batch,is_pred=True,*kwargs)
        return output

    def predict_sliding(self,img,stride=40,argmax=True):
        img_size = img.shape[0:2]
        if img_size[0] > self.h:
            rows_num = int(math.ceil((img_size[0] - self.h)/stride))
        else:
            rows_num = 1
        if img_size[1] > self.w:
            cols_num = int(math.ceil((img_size[1] - self.w)/stride))
        else:
            cols_num = 1
        full = np.zeros((img_size[0],img_size[1],self.category_num))
        sliding_num = 0
        for row in range(rows_num):
            for col in range(cols_num):
                w1 = int(col*stride)
                h1 = int(row*stride)

                crop_img = img[h1:h1+self.h,w1:w1+self.w,:]
                crop_img_size = crop_img.shape[0:2]
                crop_img = np.pad(crop_img,((0,self.h-crop_img_size[0]),(0,self.w-crop_img_size[1]),(0,0)),mode="constant",constant_values=((0,0),(0,0),(0,0)))
                crop_output = self.predict_(crop_img)
                #crop_output = ndimage.interpolation.zoom(crop_output,[img_size[0]/crop_output.shape[0],img_size[0]/crop_output.shape[1],1])
                full[h1:h1+crop_img_size[0],w1:w1+crop_img_size[1],:] += crop_output
                sliding_num += 1
        full /= sliding_num
        if argmax is True:
            full = np.argmax(full,axis=2)
        return full

    def predict_multi_scale(self,img,scales=[1],stride=10,argmax=True):
        img_size = img.shape[0:2]
        full = np.zeros((img_size[0],img_size[1],self.category_num))
        for scale in scales:
            scaled_img = 255*imgtf.rescale(img/255,scale)
            scaled_output = self.predict_sliding(scaled_img,stride=stride,argmax=False)
            scaled_output = ndimage.interpolation.zoom(scaled_output,[img_size[0]/scaled_output.shape[0],img_size[0]/scaled_output.shape[1],1]) # if the scale is not 1
            full += scaled_output
        full /= len(scales)
        if argmax is True:
            full = np.argmax(full,axis=2)
        return full

    def metrics_predict_tf(self,category="val",save_pred=False):
        self.data.reset_info()
        cur_epoch = self.data.get_cur_epoch(category)
        start_time = time.time()
        i = 0
        label_rate = math.ceil(255 / self.category_num)
        while cur_epoch == self.data.get_cur_epoch(category):
        #while i<10:
            if i % 50 == 0:
                print("start %d ..." % i)
            x,y,_ = self.data.next_batch(batch_size=1,category=category)
            #print("x shape:%s y shape:%s" % (str(x.shape),str(y.shape)))
            feed_dict = {self.net["input"]:x,self.net["label"]:y,self.net["prob"]:1.0}
            confucion_matrix = self.sess.run(self.update_op,feed_dict=feed_dict)
            i+=1
        print("tf miou:%f" % self.mIoU.eval(session=self.sess))
        hist = np.reshape(confucion_matrix,[self.category_num,self.category_num])
        m = metrics_np(n_class=self.category_num,hist=hist)
        print("all metrics:%s" % str(m.get_all()))


    def metrics_predict(self,category="val",scales=[1],save_pred=False,label_rate=True):
        self.data.reset_info()
        cur_epoch = self.data.get_cur_epoch(category)
        i = 0
        label_rate = math.ceil(255 / self.category_num)
        if label_rate is False:
            label_rate = 1
        m = metrics_np(n_class=self.category_num)
        while cur_epoch == self.data.get_cur_epoch(category):
            if i % 50 == 0:
                print("start %d ... " % i)
            x,y,_ = self.data.next_batch(category=category,batch_size=1)
            x = np.squeeze(x,axis=0)
            y = np.squeeze(y,axis=0)
            y_output = self.predict_multi_scale(x,scales=scales,argmax=False)
            if self.crf_config is not None:
                y_no_crf = np.argmax(y_output,axis=2)
                x_ = (np.reshape(x,(self.h,self.w,3))+self.data.img_mean).astype(np.uint8)
                y_output = y_output.astype(np.float32)
                y_ = crf_inference(y_output,x_,self.crf_config,self.category_num)
            else:
                y_ = np.argmax(y_output,axis=2)
            m.update(y,y_)
            i += 1
            if save_pred is True:
                imgio.imsave(os.path.join("preds","%d_origin.png" % i),(np.reshape(x,(self.input_size[0],self.input_size[1],3))+self.data.img_mean)/255.0) # the x is float of [0,255]
                imgio.imsave(os.path.join("preds","%d_gt.png" % i),np.reshape(y,(self.input_size[0],self.input_size[1]))*label_rate) # the y is ubyte of [0,255]
                imgio.imsave(os.path.join("preds","%d_pred.png" % i),np.reshape(y_,(self.input_size[0],self.input_size[1]))*label_rate) # the y is ubyte of [0,255]
                np.save(os.path.join("preds","%d_output.npy" % i),y_output)
                if self.crf_config is not None:
                    imgio.imsave(os.path.join("preds","%d_no_crf_pred.png" % i),np.reshape(y_no_crf,(self.input_size[0],self.input_size[1]))*label_rate) # the y is ubyte of [0,255]

        ret_metrics = m.get_all()
        print("metrics:%s" % str(ret_metrics))
        return ret_metrics

    def test(self):
        #img = np.ones((1,240,240,3))
        #print("predict_")
        #self.predict_(img)
        #print("predict")
        #self.predict(img)
        #img = np.ones((2,400,400,3))
        #print("predict_sliding")
        #self.predict_sliding(img,stride=100)
        #print("predict_multi_scale")
        #self.predict_multi_scale(img,scales=[1,1.3])
        metrics = self.metrics_predict(scales=[1])
        print("metrics:%s" % str(metrics))

def test_crf(): # you need to comment the self.load_saver in __init__
    img = np.ones((4,4,3),dtype=np.uint8)
    f = np.ones((4,4,2),dtype=np.float32)
    f[0,:,:] *= 0.3
    f[1,:,:] *= 0.7
    f[0,0,0] = 0.9
    f[1,0,0] = 0.1
    config = {"input_size":(4,4),"category_num":2,"crf":{"g_sxy":10,"g_compat":10,"bi_sxy":10,"bi_srgb":10,"bi_compat":10,"iterations":5}}
    p = Predict(config=config)
    print("return: %s" % str(p.crf_inference(f,img)))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    #test_crf()
    input_size = (321,321)
    crf_config = {"bi_sxy":121,"bi_srgb":5,"bi_compat":10,"g_sxy":3,"g_compat":3,"iterations":10}
    data = dataset_np({"input_size":input_size,"categorys":["val"]})
    #data = dataset_tf({"input_size":input_size})

    p = Predict(config = {"input_size":input_size,"saver_path":sys.argv[2],"data":data,"crf":crf_config})
    #p = Predict(config = {"input_size":input_size,"saver_path":sys.argv[2],"data":data})

    start_time = time.time()
    #p.metrics_predict_tf(save_pred=False)
    #start_time = time.time()
    p.metrics_predict(save_pred=False)
    #p.metrics_predict(save_pred=False,scales=[1,1.2,1.4])
    #p.metrics_predict(save_pred=False,scales=[1,1.4])
    end_time = time.time()
    print("total time:%f" % (end_time - start_time))
