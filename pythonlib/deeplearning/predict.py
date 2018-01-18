import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
from scipy import ndimage
import skimage.io as imgio
from dataset import dataset_np as dataset
from metrics import metrics_np
import skimage.transform as imgtf
import pydensecrf.densecrf as dcrf

class Predict():
    def __init__(self,config):
        self.config = config
        self.crf_config = config.get("crf",None)
        self.categorys_num = self.config.get("categorys_num",21)
        self.input_size = self.config.get("input_size",(240,240)) # (w,h)
        self.h,self.w = self.input_size
        self.sess = tf.Session()
        self.net = {}
        self.data = self.config.get("data",None)
        self.load_saver(self.config.get("saver_path","saver/saver"))

    def load_saver(self,saver_filename):
        sess = self.sess
        saver = tf.train.import_meta_graph("%s.meta" % saver_filename)
        self.net["input"] = tf.get_collection("input")[0]
        self.net["drop_probe"] = tf.get_collection("drop_probe")[0]
        self.net["is_training"] = tf.get_collection("is_training")[0]
        self.net["output"] = tf.get_collection("output")[0]
        self.net["label"] = tf.placeholder(tf.uint8,[None,self.h,self.w])
        self.net["pred"] = tf.get_collection("pred")[0]
        self.mIoU, self.update_op = tf.metrics.mean_iou(self.net["label"],self.net["pred"],num_classes=self.categorys_num)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess,saver_filename)

    def predict_(self,input,single_batch=True,**kwargs):
        params = {self.net["input"]:input,self.net["is_training"]:False,self.net["drop_probe"]:1.0}
        for key in kwargs:
            params[self.net[key]] = kwargs[key]
        if is_pred is True:
            output = self.sess.run(self.net["pred"],feed_dict=params)
        else:
            output = self.sess.run(self.net["output"],feed_dict=params)
        if single_batch is True:
            output = np.squeeze(output,axis=0)

        return output # note: after softmax, before argmax

    def predict(self,img,single_batch=True,**kwargs):
        output = self.predict_(img,single_batch=single_batch,is_pred=True,*kwargs)
        return output

    #def predict_sliding_batch(self,imgs,stride=10,argmax=True):
    def predict_sliding_batch(self,imgs,stride=10,argmax=True):
        img_size = imgs.shape[1:3]
        batch_size = imgs.shape[0]
        if img_size[0] > self.h:
            rows_num = int(math.ceil((img_size[0] - self.h)/stride))
        else:
            rows_num = 1
        if img_size[1] > self.w:
            cols_num = int(math.ceil((img_size[1] - self.w)/stride))
        else:
            cols_num = 1
        full = np.zeros((batch_size,img_size[0],img_size[1],self.categorys_num))
        sliding_num = 0
        for row in range(rows_num):
            for col in range(cols_num):
                w1 = int(col*stride)
                h1 = int(row*stride)

                crop_img = imgs[:,h1:h1+self.h,w1:w1+self.w,:]
                crop_img_size = crop_img.shape[1:3]
                #print("crop img shape:%s" % str(crop_img.shape))
                crop_img = np.pad(crop_img,((0,0),(0,self.h-crop_img_size[0]),(0,self.w-crop_img_size[1]),(0,0)),mode="constant",constant_values=((0,0),(0,0),(0,0),(0,0)))
                crop_output = self.predict_(np.reshape(crop_img,[batch_size,self.h,self.w,3]),single_batch=False)
                # need to resize to origin size

                full[:,x1:x1+crop_img_size[0],y1:y1+crop_img_size[1],:] += crop_output
                sliding_num += 1
        #print("sliding_num:%d" % sliding_num)
        full /= sliding_num
        if argmax is True:
            full = np.argmax(full,axis=3)
        return full

    def predict_multi_scale_batch(self,imgs,scales=[1],stride=10,argmax=True):
        img_size = imgs.shape[1:3]
        batch_size = imgs.shape[0]
        full = np.zeros((batch_size,img_size[0],img_size[1],self.categorys_num))
        for scale in scales:
            scaled_imgs = np.expand_dims(imgtf.rescale(imgs[0,:,:,:],scale),axis=0)
            for i in range(1,batch_size):
                scaled_imgs = np.concatenate([scaled_imgs,np.expand_dims(imgtf.rescale(imgs[i,:,:,:],scale),axis=0)],axis=0)
            scaled_output = self.predict_sliding_batch(scaled_imgs,stride=stride,argmax=False)
            outputs = np.expand_dims(imgtf.resize(scaled_output[0,:,:,:],img_size),axis=0)
            for i in range(1,batch_size):
                outputs = np.concatenate([outputs,np.expand_dims(imgtf.resize(scaled_output[i,:,:,:],img_size),axis=0)],axis=0)
            full += outputs
        full /= len(scales)
        if argmax is True:
            full = np.argmax(full,axis=3)
        return full

    def miou_predict_tf(self,category="val",scales=[1],save_pred=False):
        self.data.reset_info()
        cur_epoch = self.data.get_cur_epoch(category)
        start_time = time.time()
        i = 0
        label_rate = math.ceil(255 / self.categorys_num)
        while cur_epoch == self.data.get_cur_epoch(category):
            if i % 10 == 0:
                print("start %d ..." % i)
            x,y = self.data.next_batch(batch_size=1,category=category)
            #print("x shape:%s y shape:%s" % (str(x.shape),str(y.shape)))
            feed_dict = {self.net["input"]:x,self.net["label"]:y,self.net["drop_probe"]:1.0}
            i+=1
            if save_pred is True:
                y_ = np.argmax(y_,axis=3)
                imgio.imsave(os.path.join("pascal_voc","preds","%d_origin.png" % i),(np.reshape(x,(self.h,self.w,3))+self.data.saver_info["mean_value"])/255.0) # the x is float of [0,255]
                imgio.imsave(os.path.join("pascal_voc","preds","%d_gt.png" % i),np.reshape(y,(self.h,self.w))*label_rate) # the y is ubyte of [0,255]
                imgio.imsave(os.path.join("pascal_voc","preds","%d_pred.png" % i),np.reshape(y_,(self.h,self.w))*label_rate) # the y is ubyte of [0,255]
            else:
                self.sess.run(self.update_op,feed_dict=feed_dict)
        print("miou:%f" % self.mIoU.eval(session=self.sess))


    def metrics_predict(self,category="val",scales=[1.2,1.6],save_pred=False):
        self.data.reset_info()
        cur_epoch = self.data.get_cur_epoch(category)
        iou = [0 for i in range(self.categorys_num)]
        len_iou = [0 for i in range(self.categorys_num)]
        i = 0
        label_rate = math.ceil(255 / self.categorys_num)
        all_labels,all_preds = [],[]
        start_time = time.time()
        while cur_epoch == self.data.get_cur_epoch(category):
            if i % 10 == 0:
                print("start %d ... " % i)
            x,y = self.data.next_batch(category=category,batch_size=1)
            if self.crf_config is not None:
                y_origin = self.predict_multi_scale_batch(x,scales=scales,argmax=False)
                y_origin = y_origin.astype(np.float32)
                x = np.reshape(x,(self.h,self.w,3)).astype(np.uint8)
                y_ = self.crf_inference(y_origin,x)
            else:
                y_ = self.predict_multi_scale_batch(x,scales=scales)
            all_labels.append(y)
            all_preds.append(y_)
            i += 1
            if save_pred is True:
                imgio.imsave(os.path.join("pascal_voc","preds","%d_origin.png" % i),(np.reshape(x,(self.input_size[0],self.input_size[1],3)))/255.0) # the x is float of [0,255]
                imgio.imsave(os.path.join("pascal_voc","preds","%d_gt.png" % i),np.reshape(y,(self.input_size[0],self.input_size[1]))*label_rate) # the y is ubyte of [0,255]
                imgio.imsave(os.path.join("pascal_voc","preds","%d_pred.png" % i),np.reshape(y_,(self.input_size[0],self.input_size[1]))*label_rate) # the y is ubyte of [0,255]
                np.save(os.path.join("pascal_voc","preds","%d_before_crf.npy" % i),y_origin)
                y_origin = np.argmax(y_origin,axis=3)
                imgio.imsave(os.path.join("pascal_voc","preds","%d_no_crf_pred.png" % i),np.reshape(y_origin,(self.input_size[0],self.input_size[1]))*label_rate) # the y is ubyte of [0,255]

        ret_metrics = metrics_np(all_labels,all_preds,self.categorys_num)
        end_time = time.time()
        print("total time:%f" % (end_time - start_time))
        print("metrics:%s" % str(ret_metrics))
        return ret_metrics

    def crf_inference(self,feat,img):
        '''
        feat: the feature map of cnn, shape [h,w,c]
        img: the origin img, shape [h,w,3]
        '''
        assert self.crf_config is not None, "the config of crf is not given"
        crf = dcrf.DenseCRF2D(self.h,self.w,self.categorys_num)

        # unary energy
        unary = feat.reshape((self.categorys_num,-1))
        crf.setUnaryEnergy(unary)

        # pairwise energy
        crf.addPairwiseGaussian( sxy=self.crf_config["g_sxy"], compat=self.crf_config["g_compat"] )
        crf.addPairwiseBilateral( sxy=self.crf_config["bi_sxy"], srgb=self.crf_config["bi_srgb"], rgbim=img, compat=self.crf_config["bi_compat"] )
        Q = crf.inference( self.crf_config["iterations"] )
        r = np.argmax(Q,axis=0).reshape((self.h,self.w))
        return r
        
    def test(self):
        #img = np.ones((1,240,240,3))
        #print("predict_")
        #self.predict_(img)
        #print("predict")
        #self.predict(img)
        #img = np.ones((2,400,400,3))
        #print("predict_sliding")
        #self.predict_sliding_batch(img,stride=100)
        #print("predict_multi_scale")
        #self.predict_multi_scale_batch(img,scales=[1,1.3])
        metrics = self.metrics_predict(scales=[1])
        print("metrics:%s" % str(metrics))

def test_crf(): # you need to comment the self.load_saver in __init__
    img = np.ones((4,4,3),dtype=np.uint8)
    f = np.ones((4,4,2),dtype=np.float32)
    f[0,:,:] *= 0.3
    f[1,:,:] *= 0.7
    f[0,0,0] = 0.9
    f[1,0,0] = 0.1
    config = {"input_size":(4,4),"categorys_num":2,"crf":{"g_sxy":10,"g_compat":10,"bi_sxy":10,"bi_srgb":10,"bi_compat":10,"iterations":5}}
    p = Predict(config=config)
    print("return: %s" % str(p.crf_inference(f,img)))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    #test_crf()
    input_size = (321,321)
    data = dataset({"input_size":input_size})
    crf_config = {"bi_sxy":121,"bi_srgb":5,"bi_compat":10,"g_sxy":3,"g_compat":3,"iterations":5}
    p = Predict(config = {"input_size":input_size,"saver_path":sys.argv[2],"data":data,"crf":crf_config})
    #p.miou_predict_tf(save_pred=False)
    p.metrics_predict(save_pred=True)
