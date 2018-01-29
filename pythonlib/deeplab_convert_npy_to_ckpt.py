import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

import tensorflow as tf
import numpy as np
from network import Network
from estep import estep as _estep
from dataset import dataset_tf as dataset
from metrics import metrics_update
import time

class ADAPT(Network):
    def __init__(self,config):
        Network.__init__(self,config)
        self.stride = {}
        self.stride["input"] = 1

        # different lr for different variable
        self.lr_1_list = []
        self.lr_2_list = []
        self.lr_10_list = []
        self.lr_20_list = []

    def build(self):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = tf.placeholder(tf.float32,[None,self.h,self.w,self.config.get("input_channel",3)])
                self.net["label"] = tf.placeholder(tf.float32,[None,self.h,self.w,1])
                self.net["drop_probe"] = tf.placeholder(tf.float32)

            self.net["output"] = self.create_network()
            self.pred()
        return self.net["output"]

    def create_network(self):
        if "init_model_path" in self.config:
            self.load_init_model()
        with tf.name_scope("vgg") as scope:
            # build block
            block = self.build_block("input",["conv1_1","relu1_1","conv1_2","relu1_2","pool1"])
            block = self.build_block(block,["conv2_1","relu2_1","conv2_2","relu2_2","pool2"])
            block = self.build_block(block,["conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3"])
            block = self.build_block(block,["conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4"])
            block = self.build_block(block,["conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5"])
            fc = self.build_fc(block,["fc6","relu6","drop6","fc7","relu7","drop7","fc8"])

            # classifier
            return tf.nn.softmax(self.net[fc])

    def build_block(self,last_layer,layer_lists):
        for layer in layer_lists:
            if layer.startswith("conv"):
                if layer[4] != "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
                if layer[4] == "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.relu( self.net[last_layer],name="relu")
                    last_layer = layer
            if layer.startswith("pool"):
                if layer[4] not in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = 2 * self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME",name="pool")
                        last_layer = layer
                if layer[4] in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                        last_layer = layer
        return last_layer

    def build_fc(self,last_layer, layer_lists):
        for layer in layer_lists:
            if layer.startswith("fc"):
                with tf.name_scope(layer) as scope:
                    weights,bias = self.get_weights_and_bias(layer)
                    if last_layer.startswith("pool"):
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=4, padding="SAME", name="conv")

                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.relu( self.net[last_layer])
                    last_layer = layer
            if layer.startswith("drop"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.dropout( self.net[last_layer],self.net["drop_probe"])
                    last_layer = layer

        return last_layer

    def pred(self):
        scale_output = tf.image.resize_bilinear(self.net["output"],self.net["input"].shape[1:3])
        self.net["pred"] = tf.argmax(scale_output,axis=3)

    def e_step(self,last_layer, bg_p, fg_p, num_iter, suppress_others, margin_others):
        shrink_label = tf.squeeze(tf.image.resize_nearest_neighbor(self.net["label"],self.net["output"].shape[1:3]),axis=3)
        def estep(feature_map,label):
            s = time.time()
            #print("start time:%f" % s)
            tmp_ = _estep(feature_map,label,suppress_others,num_iter,margin_others,bg_p,fg_p,use_c=True)
            #tmp_ = _estep(feature_map,label,suppress_others,num_iter,margin_others,bg_p,fg_p)
            e = time.time()
            #print("duration time :%f " % (e-s))
            return tmp_
        layer = "e_step"
        self.net[layer] = tf.py_func(estep,[self.net[last_layer],shrink_label],tf.float32)
        last_layer = layer
        layer = "e_argmax"
        self.net[layer] = tf.argmax(self.net[last_layer],axis=3)
        return layer

    def load_init_model(self):
        model_path = self.config["init_model_path"]
        self.init_model = np.load(model_path,encoding="latin1").item()
        print("load init model success: %s" % model_path)

    def get_weights_and_bias(self,layer):
        print("layer: %s" % layer)
        if layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1":
                shape[2] = 3
            else:
                shape[2] = 64 * self.stride[layer]
                if shape[2] > 512: shape[2] = 512
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2] = int(shape[2]/2)
            shape[3] = 64 * self.stride[layer]
            if shape[3] > 512: shape[3] = 512
        if layer.startswith("fc"):
            if layer == "fc6":
                shape = [4,4,512,4096]
            if layer == "fc7":
                shape = [1,1,4096,4096]
            if layer == "fc8": 
                shape = [1,1,4096,self.category_num]
        if layer == "fc8":
            init = tf.constant_initializer(self.init_model["fc8_voc12"]["w"])
        else:
            init = tf.constant_initializer(self.init_model[layer]["w"])
        weights = tf.get_variable(name="%s_weights" % layer,initializer=init,shape = shape)
        if layer == "fc8":
            init = tf.constant_initializer(self.init_model["fc8_voc12"]["b"])
        else:
            init = tf.constant_initializer(self.init_model[layer]["b"])
        bias = tf.get_variable(name="%s_bias" % layer,initializer=init,shape = [shape[-1]])
        self.weights[layer] = (weights,bias)
        if layer != "fc8":
            self.lr_1_list.append(weights)
            self.lr_2_list.append(bias)
        else:
            self.lr_10_list.append(weights)
            self.lr_20_list.append(bias)

        return weights,bias

    def getloss(self):
        weaklabel_layer = self.e_step("fc8", bg_p=0.4, fg_p=0.2, num_iter=5, suppress_others=True, margin_others=1e-5)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.net[weaklabel_layer],[-1]),logits=tf.reshape(self.net["output"],[-1,self.category_num])))
        return loss

    def optimize(self,base_lr,momentum):
        self.net["lr"] = tf.Variable(base_lr, trainable=False)
        opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        gradients = opt.compute_gradients(self.loss["total"])
        gradients = [(g,v) for (g,v) in gradients if g is not None]
        a = tf.Variable(1.0,dtype=tf.float32)
        for (g,v) in gradients:
            if v in self.lr_2_list:
                g = 2*g
            if v in self.lr_10_list:
                g = 10*g
            if v in self.lr_20_list:
                g = 20*g
                print("20 x gradient")
            
            a = tf.Print(a,[g.name,tf.reduce_mean(g)],"gradient")
            a = tf.Print(a,[v.name,tf.reduce_mean(v)],"weight")
            a = tf.Print(a,[v.name,tf.reduce_mean(g)/(tf.reduce_mean(v)+1e-20)],"rate")
        self.net["train_op"] = opt.apply_gradients(gradients)
        self.net["g"] = a

    def train(self,base_lr,weight_decay,momentum,batch_size,epoches):
        assert self.data is not None,"data is None"
        assert self.sess is not None,"sess is None"
        self.net["is_training"] = tf.placeholder(tf.bool)
        x_train,y_train,iterator_train = self.data.next_data(category="train",batch_size=batch_size,epoches=-1)
        x_val,y_val,iterator_val = self.data.next_data(category="val",batch_size=batch_size,epoches=-1)
        x = tf.cond(self.net["is_training"],lambda:x_train,lambda:x_val)
        y = tf.cond(self.net["is_training"],lambda:y_train,lambda:y_val)
        self.build()
        self.pre_train(base_lr,weight_decay,momentum,batch_size,save_layers=["input","output","label","pred","is_training","drop_probe"])

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            self.sess.run(iterator_val.initializer)

            print("start to save model: %s" % self.config.get("save_model_path"))
            self.saver["norm"].save(self.sess,self.config.get("save_model_path"))
            print("saved model: %s" % self.config.get("save_model_path"))

if __name__ == "__main__":
    batch_size = 8
    input_size = (321,321)
    category_num = 21
    epoches = 10
    data = dataset({"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num})
    adapt = ADAPT({"data":data,"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"init_model_path":"./model/weak_train2_iter_8000.npy","save_model_path":"model/weak_train2_iter_8000.ckpt"})
    adapt.train(base_lr=0.001,weight_decay=5e-4,momentum=0.7,batch_size=batch_size,epoches=epoches)
