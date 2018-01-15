import tensorflow as tf
import numpy as np
from network import Network

class VGG16(Network):
    def __init__(self,config):
        Network.__init__(self,config)
        self.stride = {}

    def build(self):
        with tf.name_scope("vgg") as scope:
            self.net["input"] = tf.placeholder(tf.float32, [None, self.config["input_h"],self.config["input_w"],3])
            self.net["drop_probe"] = tf.placeholder(tf.float32)
            self.stride["input"] = 1

            # build block
            block = self.build_block("input",["conv1_1","relu1_1","conv1_2","relu1_2","pool1"])
            block = self.build_block(block,["conv2_1","relu2_1","conv2_2","relu2_2","pool2"])
            block = self.build_block(block,["conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3"])
            block = self.build_block(block,["conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4"])
            block = self.build_block(block,["conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5"])
            fc = self.build_fc(block,["fc6","relu6","drop6","fc7","relu7","drop7","fc8"])

            # classifier
            self.net["output"] = tf.nn.softmax(self.net[fc])

    def build_block(self,last_layer,layer_lists):
        for layer in layer_lists:
            if layer.startswith("conv"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    weights,bias = self.get_weights_and_bias(layer)
                    self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                    self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.relu( self.net[last_layer],name="relu")
                    last_layer = layer
            if layer.startswith("pool"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = 2 * self.stride[last_layer]
                    self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME",name="pool")
                    last_layer = layer
        return last_layer

    def build_fc(self,last_layer, layer_lists):
        for layer in layer_lists:
            if layer.startswith("fc"):
                with tf.name_scope(layer) as scope:
                    weights,bias = self.get_weights_and_bias(layer)
                    if last_layer.startswith("pool"):
                        self.net[layer] = tf.matmul(tf.reshape(self.net[last_layer],shape=[-1,7*7*512]), weights)+ bias
                    else:
                        self.net[layer] = tf.matmul(self.net[last_layer], weights) + bias
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
                shape = [7*7*512,4096]
            if layer == "fc7":
                shape = [4096,4096]
                layer = "hidden_output"
            if layer == "fc8": 
                shape = [4096,1000]
                layer = "output"
        if "trained_model" not in self.config:
            weights = tf.get_variable(name="%s_weights" % layer,shape = shape)
            bias = tf.get_variable(name="%s_bias" % layer, shape = [shape[-1]])
        else:
            init = tf.constant_initializer(self.trained_model[layer]["weights"])
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init,shape = shape)
            init = tf.constant_initializer(self.trained_model[layer]["biases"])
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init,shape = [shape[-1]])
        return weights,bias

    def tensorboard(self):
        sess = tf.Session()
        tf.summary.scalar("output",tf.reduce_sum(self.net["output"]))
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/",sess.graph)
        sess.run(tf.global_variables_initializer())
        a = sess.run(merged,feed_dict={self.net["input"]:np.ones([1,self.config["input_h"],self.config["input_w"],3]),self.net["drop_probe"]:0.5})
        writer.add_summary(a)

if __name__ == "__main__":
    vgg = VGG16(config = {"trained_model":"trained_model/VGG_16.npy"})
    vgg.build()
    vgg.tensorboard()
