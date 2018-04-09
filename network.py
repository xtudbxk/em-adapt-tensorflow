import time
import numpy as np
import tensorflow as tf

from dataset import dataset

class Network():
    def __init__(self,config):
        self.config = config
        self.h,self.w = self.config.get("input_size",(25,25))
        self.category_num = self.config.get("category_num",21)
        self.accum_num = self.config.get("accum_num",1)
        self.data = self.config.get("data",None)
        #self.sess = self.config.get("sess",tf.Session())
        self.net = {}
        self.weights = {}
        self.trainable_list = []
        self.loss = {}
        self.images = {}
        self.saver = {}

    def build(self):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = tf.placeholder(tf.float32,[None,self.h,self.w,self.config.get("input_channel",3)])
                self.net["label"] = tf.placeholder(tf.float32,[None,self.h,self.w,self.config.get("output_channel",1)])

            self.net["output"] = self.create_network()
            self.pred()
        return self.net["output"]

    # need to rewrite
    def create_network(self,layer):
        if "init_model_path" in self.config:
            self.load_init_model()
        return layer # note no softmax

    # need to rewrite
    def pred(self):
        self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"],(self.h,self.w))
        self.net["pred"] = tf.argmax(self.net["rescale_output"],axis=3)

    # need to rewrite
    def load_init_model(self):
        pass

    # need to rewrite
    def get_weights_and_biases(self,layer):
        if layer in self.weights:
            return self.weights[layer]
        w,b = None,None
        self.weights[layer] = (w,b)
        self.trainable_list.append(w)
        self.trainable_list.append(b)
        return w,b
    
    def predict(self):
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))
        self.sess = tf.Session(config=gpu_options)
        self.build()
        self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)

        crf_config = {"bi_sxy":121,"bi_srgb":5,"bi_compat":10,"g_sxy":3,"g_compat":3,"iterations":10}
        #crf_config = None
        data = dataset_np({"input_size":self.config.get("input_size"),"categorys":["val"]}) # this is not same with self.data
        config = {"input_size":self.config.get("input_size"),"crf":crf_config,"sess":self.sess,"net":self.net,"data":data}
        p = Predict(config,create_net=False)

        if self.config.get("model_path",False) is not False:
            print("start to load model: %s" % self.config.get("model_path"))
            self.restore_from_model(self.saver["norm"],self.config.get("model_path"),checkpoint=False)
            print("model loaded ...")
        start_time = time.time()
        end_time = time.time()
        print("total time:%f" % (end_time - start_time))

    def remove_ignore_label(self,gt,output): 
        ''' 
        gt: not one-hot 
        output: a distriution of all labels, and is scaled to macth the size of gt
        NOTE the result is a flatted tensor
        and all label which is bigger that or equal to self.category_num is void label
        '''
        gt = tf.reshape(gt,shape=[-1])
        output = tf.reshape(output, shape=[-1,self.category_num])
        indices = tf.squeeze(tf.where(tf.less(gt,self.category_num)),axis=1)
        gt = tf.gather(gt,indices)
        output = tf.gather(output,indices)
        return gt,output

    def pre_train(self,base_lr,weight_decay,momentum,batch_size,save_layers=["input","output","label","pred"]):
        self.net["accum_gradient"] = [tf.Variable(tf.zeros_like(v),trainable=False) for v in self.trainable_list]
        with self.sess.as_default():
            self.getloss(weight_decay)
            self.optimize(base_lr,momentum)

            for layer in save_layers:
                tf.add_to_collection(layer,self.net[layer])

            self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)
            self.saver["lr"] = tf.train.Saver(var_list=self.trainable_list)
            self.saver["best"] = tf.train.Saver(var_list=self.trainable_list,max_to_keep=2)

    def restore_from_model(self,saver,model_path,checkpoint=False):
        assert self.sess is not None
        if checkpoint is True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, model_path)
