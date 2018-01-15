import tensorflow as tf
import numpy as np
from network import Network

class ResNet50(Network):
    # note this class need care for the batch norm update option  while traininn
    def __init__(self,config):
        Network.__init__(self,config)

    def build(self):
        with tf.name_scope("resnet") as scope:
            self.net["input"] = tf.placeholder(tf.float32, [None, self.config["input_h"], self.config["input_w"],3])
            self.net["is_training"] = tf.placeholder(tf.bool)

            # build common prefix layers
            stack = self.build_prefix_layers("input")

            # build stack
            stack = self.build_stack(stack,"2",3,False)
            stack = self.build_stack(stack,"3",4,True)
            stack = self.build_stack(stack,"4",6,True)
            stack = self.build_stack(stack,"5",3,True)
            
            # build post layer
            fc = self.build_post_layer(stack)

            # classifier
            self.net["output"] = tf.nn.softmax(self.net[fc])

    def build_prefix_layers(self, last_layer):
        with tf.name_scope("prefix_layer") as scope:
            layer = "conv1"
            weights,bias = self.get_weights_bias(layer)
            self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,2,2,1], padding="SAME",name=layer) + bias
            last_layer  = layer
            layer = "bn_conv1"
            self.net[layer] = tf.contrib.layers.batch_norm( self.net[last_layer])
            last_layer  = layer
            layer = "conv1_relu"
            self.net[layer] = tf.nn.relu( self.net[last_layer],name=layer)
            last_layer = layer
            layer = "pool1"
            self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME",name=layer)
            return layer

    def build_stack(self,last_layer,stack_identifer,block_num,is_first_block_pool = True):
        block_identifers = ["a","b","c","d","e","f","g","h"]
        block_identifers = block_identifers[:block_num]
        with tf.name_scope("stack%s" % stack_identifer) as scope:
            for i in block_identifers:
                if i == "a":
                    last_layer = self.build_block(last_layer, stack_identifer,i,is_first_block_pool)
                else:
                    last_layer = self.build_block(last_layer, stack_identifer,i)
        return last_layer

    def build_block(self,last_layer,stack_identifer,block_identifer,pool = False):
        with tf.name_scope("block%s" % block_identifer) as scope:
            shortcut = last_layer
            for i in ["a","b","c"]:
                with tf.name_scope("conv%s" % i) as scope:
                    # conv
                    layer = "res%s%s_branch2%s" % (stack_identifer,block_identifer,i)
                    weights,bias = self.get_weights_bias(layer)
                    if i == "a" and pool is True:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,2,2,1], padding="SAME",name=layer) + bias
                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME",name=layer) + bias
                    last_layer = layer 
                    # batch norm
                    layer = "bn%s%s_branch2%s" % (stack_identifer,block_identifer,i)
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer],is_training=self.net["is_training"])
                    last_layer = layer
                    # relu
                    if i != "c":
                        layer = "res%s%s_branch2%s_relu" % (stack_identifer,block_identifer,i)
                        self.net[layer] = tf.nn.relu(self.net[last_layer],name=layer)
                        last_layer = layer
            with tf.name_scope("shortcut") as scope:
                if block_identifer == "a":
                    layer = "res%sa_branch1" % stack_identifer
                    weights,bias = self.get_weights_bias(layer)
                    if pool is True:
                        self.net[layer] = tf.nn.conv2d( self.net[shortcut], weights, strides = [1,2,2,1], padding="SAME",name=layer) + bias
                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[shortcut], weights, strides = [1,1,1,1], padding="SAME",name=layer) + bias
                    shortcut = layer
                    layer = "bn%sa_branch1" % stack_identifer
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[shortcut],is_training=self.net["is_training"])
                    shortcut = layer
            layer = "res%s%s" % (stack_identifer,block_identifer)
            self.net[layer] = self.net[last_layer] + self.net[shortcut]
            last_layer = layer
            layer = "res%s%s_relu" % (stack_identifer,block_identifer)
            self.net[layer] = tf.nn.relu(self.net[last_layer],name=layer)
            return layer

    def build_post_layer(self,last_layer):
        with tf.name_scope("post_layer") as scope:
            layer = "pool5"
            self.net[layer] = tf.nn.avg_pool(self.net[last_layer], ksize=[1,7,7,1], strides=[1,1,1,1], padding="VALID",name=layer)
            last_layer = layer
            layer = "fc1000"
            weights,bias = self.get_weights_bias(layer)
            self.net[layer] = tf.matmul(tf.reshape(self.net[last_layer],shape=[-1,2048]),weights,name=layer)+bias
            return layer
    #def batch_norm_layer(self,input,beta=None,gamma=None,):
        #beta = tf.Variable(


    def get_weights_bias(self,layer):
        if layer.startswith("conv1"): # the prefix layer
            shape = [7,7,3,64]
        if layer.startswith("fc"): # the post layer
            shape = [2048,1000]
        if layer.startswith("res"): # 
            stack_index = int(layer[3])
            block_index = layer[4]
            base = 16 * 2**stack_index
            if "branch1" in layer:
                if stack_index == 2:
                    shape = [1,1,64,4*base]
                else:
                    shape = [1,1,2*base,4*base]
            elif layer.endswith("branch2a"):
                if block_index == "a":
                    if stack_index == 2:
                        shape = [1,1,base,base]
                    else:
                        shape = [1,1,2*base,base]
                else:
                    shape = [1,1,4*base,base]
            elif layer.endswith("branch2b"): # the 3x3 conv
                shape = [3,3,base,base]
            elif layer.endswith("branch2c"):
                shape = [1,1,base,4*base]
        print("layer: %s" % layer)
        print("layer: %s shape:%s" % (layer, str(shape)))
        if "trained_model" not in self.config:
            weights = tf.get_variable(name="%s_weights" % layer,shape = shape)
            bias = tf.get_variable(name="%s_bias" % layer, shape = [shape[-1]])
        else:
            init = tf.constant_initializer(self.trained_model[layer]["weights"])
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init,shape = shape)
            #init = tf.constant_initializer(self.trained_model[layer])
            #bias = tf.get_variable(name="%s_bias" % layer,initializer=init,shape = [shape[-1]])
            bias = tf.get_variable(name="%s_bias" % layer, shape = [shape[-1]])
        return weights,bias

if __name__ == "__main__":
    resnet = ResNet50(config = {"trained_model":"trained_model/res50.npy"})
    resnet.build()
    resnet.tensorboard()
