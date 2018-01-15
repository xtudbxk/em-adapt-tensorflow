import os
import sys
import math
import random
import pickle
import skimage
import numpy as np
import tensorflow as tf
import skimage.io as imgio
from datetime import datetime
import skimage.transform as imgtf

class dataset():
    def __init__(self,config={},init=True):
        self.config = config
        self.w,self.h = self.config.get("input_size",(240,240))
        self.img_mean = np.ones((self.w,self.h,3))
        self.img_mean[:,:,0] *= 104.00698793
        self.img_mean[:,:,1] *= 116.66876762
        self.img_mean[:,:,2] *= 122.67891434
        if init is True:
            self.data_f,self.data_len = self.get_data_f()
            self.data = self.get_data()
            self.info = self.reset_info()

    def get_data_f(self):
        data_f = {"train":{"x":[],"y":[]},"test":{"x":[],"y":[]},"val":{"x":[],"y":[]}}
        data_len = {"train":0,"test":0,"val":0}
        for one in ["train","val","test"]:
            with open(os.path.join("pascal_voc","txt","%s.txt" % one),"r") as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    data_f[one]["x"].append(os.path.join("pascal_voc","images","%s.jpg" % line))
                    data_f[one]["y"].append(os.path.join("pascal_voc","annotations","%s.png" % line))
            #data_f[one]["x"] = data_f[one]["x"][:100]
            #data_f[one]["y"] = data_f[one]["y"][:100]
            #print("files:%s" % str(data_f[one]["x"]))
            data_len[one] = len(data_f[one]["y"])

        print("len:%s" % str(data_len))
        return data_f,data_len

    def get_data(self):
        data = {"train":{"x":[],"y":[]},"test":{"x":[],"y":[]},"val":{"x":[],"y":[]}}
        return data

    def reset_info(self):
        info = {}
        info["epoch"] = {"train":0,"test":0,"val":0} 
        info["index"] = {"train":0,"test":0,"val":0}
        info["perm"] = {}
        perm = np.arange(self.data_len["train"])
        np.random.shuffle(perm)
        info["perm"]["train"] = perm
        perm = np.arange(self.data_len["test"])
        np.random.shuffle(perm)
        info["perm"]["test"] = perm
        perm = np.arange(self.data_len["val"])
        np.random.shuffle(perm)
        info["perm"]["val"] = perm
        return info

    def get_info(self,key="epoch",category="train"):
        return self.info[key][category]

    def get_data_len(self,category="train"):
        return self.data_len[category]

    def get_cur_epoch(self,category="train"):
        return self.info["epoch"][category]


class dataset_np(dataset):
    def __init__(self,config={},init=False):
        super(dataset_np,self).__init__(config,init=init)
        self.data_f,self.data_len = self.get_data_f()
        self.data = self.get_data()
        self.info = self.reset_info()

    def get_data(self):
        data = {"train":{"x":[],"y":[]},"test":{"x":[],"y":[]},"val":{"x":[],"y":[]}}
        #for category in ["train","test","val"]:
        for category in ["train","val"]:
            for index in range(self.data_len[category]):
                x_f = self.data_f[category]["x"][index]
                y_f = self.data_f[category]["y"][index]
                x = imgio.imread(x_f)
                x = 255* skimage.img_as_float(x)
                y = imgio.imread(y_f)
                y = skimage.img_as_ubyte(y)
                x,y = self.preprocess(x,y)
                data[category]["x"].append(x)
                data[category]["y"].append(y)
        return data

    def next_batch(self,batch_size=10,category="train",left_remove=True):
        self.info["batch_size"] = batch_size
        end_index = self.info["index"][category] + batch_size
        if end_index > self.data_len[category] and left_remove is True:
            self.info["index"][category] = 0
            end_index = batch_size
            self.info["epoch"][category] += 1
        perm = self.info["perm"][category][self.info["index"][category]:end_index]
        self.info["index"][category] = end_index
        self.info["index"][category] %= self.data_len[category]
        if end_index >= self.data_len[category]:
            self.info["epoch"][category] += 1

        x = [self.data[category]["x"][index] for index in perm]
        y = [self.data[category]["y"][index] for index in perm]
        return np.array(x),np.array(y)

    def subtract_mean(self,x): 
        return np.subtract(x,self.img_mean)

    def preprocess(self,x,y):
        w,h = x.shape[0:2]
        if w <= self.w:
            pad_width_1 = int(math.ceil((self.w - w)/2))
            pad_width_2 = int(math.floor((self.w - w)/2))
            x = np.pad(x,((pad_width_1,pad_width_2),(0,0),(0,0)),mode="constant",constant_values=0)
            y = np.pad(y,((pad_width_1,pad_width_2),(0,0)),mode="constant",constant_values=0)
            w_position = 0
        else:
            w_position = int(math.ceil((w-self.w)/2))
        if h <= self.h:
            pad_width_1 = int(math.ceil((self.h - h)/2))
            pad_width_2 = int(math.floor((self.h - h)/2))
            x = np.pad(x,((0,0),(pad_width_1,pad_width_2),(0,0)),mode="constant",constant_values=0)
            y = np.pad(y,((0,0),(pad_width_1,pad_width_2)),mode="constant",constant_values=0)
            h_position = 0
        else:
            h_position = int(math.ceil((h-self.h)/2))
        x = x[w_position:w_position+self.w,h_position:h_position+self.h,:]
        y = y[w_position:w_position+self.w,h_position:h_position+self.h]

        ##RGB -> BGR
        r,g,b = np.dsplit(x,3)
        x = np.concatenate((b,g,r),axis=2)
        x = self.subtract_mean(x)

        return x,y

    def get_histogram(self,category="train"):
        histogram = {}
        for i in range(self.data_len[category]):
            label = np.argmax(self.data[category]["y"][i])
            key = str(label)
            if key not in histogram:
                histogram[key] = 1
            else:
                histogram[key] += 1
        return histogram

class dataset_tf(dataset):
    def __init__(self,config={},init=False):
        super(dataset_tf,self).__init__(config,init)
        self.data_f,self.data_len = self.get_data_f()
        self.info = self.reset_info()

    def next_data(self,category="train",batch_size=None,epoches=-1):
        print("category:%s" % category)
        if batch_size is None:
            batch_size = self.config.get("batch_size",1)
        dataset = tf.data.Dataset.from_tensor_slices({
            "img_f":self.data_f[category]["x"],
            "label_f":self.data_f[category]["y"]
            })
        def m(x):
            img_f = x["img_f"]
            img_raw = tf.read_file(img_f)
            img = tf.image.decode_image(img_raw)
            img = tf.expand_dims(img,axis=0)
            label_f = x["label_f"]
            label_raw = tf.read_file(label_f)
            label = tf.image.decode_image(label_raw)
            label = tf.expand_dims(label,axis=0)
            if category == "train":
                img,label = self.image_preprocess(img,label,random_scale=True)
            else:
                img,label = self.image_preprocess(img,label,random_scale=False)

            img = tf.reshape(img,[self.h,self.w,3])
            label = tf.reshape(label,[self.h,self.w,1])

            return img,label

        dataset = dataset.repeat(epoches)
        dataset = dataset.shuffle(self.data_len[category])
        dataset = dataset.map(m)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        img,label = iterator.get_next()
            
        return img,label,iterator

    def get_data_tf(self,batch_size=1):
        data = {"train":{"x":None,"y":None,"name":None},"test":{"x":None,"y":None,"name":None},"val":{"x":None,"y":None,"name":None}}
        for category in ["train","val","test"]:
            print("category:%s" % category)

            images_tensor = tf.convert_to_tensor(self.data_f[category]["x"])
            labels_tensor = tf.convert_to_tensor(self.data_f[category]["y"])
            #print("f-x:%s\nf-y:%s" % (str(self.data_f[category]["x"]),str(self.data_f[category]["y"])))
            self.queues = tf.train.slice_input_producer([images_tensor,labels_tensor])

            raw_data = tf.read_file(self.queues[0])
            img_data = tf.image.decode_jpeg(raw_data)
            img_data = tf.to_float(img_data)
            raw_data = tf.read_file(self.queues[1])
            label_data = tf.image.decode_jpeg(raw_data)

            img_data, label_data = self.image_preprocess(img_data,label_data)
            
            label_data = tf.reshape(label_data,[self.w,self.h,1])
            label_data = tf.to_int32(label_data)
            print("img_data:%s" % repr(img_data))
            print("l_data:%s" % repr(label_data))
            data[category]["x"], data[category]["y"],data[category]["name"] = tf.train.batch([img_data,label_data,self.queues],batch_size,shapes=[(self.w,self.h,3),(self.w,self.h,1),(2)])
        return data

    def next_data_tf(self,category="train"):
        return self.data[category]["x"],self.data[category]["y"],self.data[category]["name"]

    def image_preprocess(self,img,label,random_scale=True,flip=False,rotate=False):
        # input img and label shape [None, h, w, c]
        if random_scale is True:
            scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
            h_new = tf.to_int32(tf.to_float(tf.shape(img)[1])* scale)
            w_new = tf.to_int32(tf.to_float(tf.shape(img)[2])* scale)
            new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])
            img = tf.image.resize_bilinear(img, new_shape)
            img = tf.squeeze(img, squeeze_dims=[0])
            label = tf.image.resize_nearest_neighbor(label, new_shape)
            label = tf.squeeze(label, squeeze_dims=[0])
        else:
            img = tf.squeeze(img, squeeze_dims=[0])
            label = tf.squeeze(label, squeeze_dims=[0])
        img = tf.image.resize_image_with_crop_or_pad(img, self.h, self.w)
        label = tf.image.resize_image_with_crop_or_pad(label, self.h, self.w)

        r,g,b = tf.split(axis=2,num_or_size_splits=3,value=img)
        img = tf.cast(tf.concat([b,g,r],2),dtype=tf.float32)
        img -= self.img_mean

        if flip is True:
            img,label = self.image_flip(img,label)

        if rotate is True:
            img,label = self.image_rotate(img,label,minangle=-math.pi/18.0, maxangle = math.pi/18.0)

        return img, label

    def image_flip(self,img,label,left_right=True, up_down=False, random_s=1):
        if left_right is True:
            r = tf.random_uniform([1])
            r = tf.reduce_sum(r)
            def flip_left_right():
                nonlocal img,label
                img = tf.image.flip_left_right(img)
                label = tf.image.flip_left_right(label)
                return 1
            tf.cond(r < random_s, flip_left_right,lambda:0)
        if up_down is True:
            r = tf.random_uniform([1])
            r = tf.reduce_sum(r)
            def flip_up_down():
                nonlocal img,label
                img = tf.image.flip_up_down(img)
                label = tf.image.flip_up_down(label)
                return 1
            tf.cond(r < random_s, flip_up_down,lambda:0)

        return img, label

    def image_rotate(self,img,label,minangle=0,maxangle=0.314):
        angle = tf.random_uniform([1],minval=minangle,maxval=maxangle)
        angle = tf.squeeze(angle,[0])
        img = tf.contrib.image.rotate(img, angle)
        label = tf.contrib.image.rotate(label,angle)
        return img, label

def test_dataset_np():
    data= dataset_np({"input_size":(240,240)})
    cur_epoch = data.get_cur_epoch(category="val")
    i = 0
    while cur_epoch == data.get_cur_epoch(category="val"):
        print("%d" % i)
        print("next_batch...")
        data.next_batch(category="val")
        i+=1

def test_dataset_tf():
    data = dataset_tf({"input_size":(321,321)})
    x,y,iterator = data.next_data(category="val",batch_size=5,epoches=1)
    #x,y,iterator = data.next_data(category="train",batch_size=4,epoches=1)
    sess = tf.Session()
    sess.run(iterator.initializer)
    i = 0
    try:
        while(True):
            if i % 10 == 0: print("i:%d" % i)
            _ = sess.run([x,y])
            i += 1
    except tf.errors.OutOfRangeError:
        print("outofrange")
    except Exception as e:
        print("exception %s" % repr(e))
    finally:
        print("finally")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    #test_dataset_np()
    test_dataset_tf()
