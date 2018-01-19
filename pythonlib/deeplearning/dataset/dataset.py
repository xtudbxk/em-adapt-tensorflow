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
    def __init__(self,config={}):
        self.config = config
        self.w,self.h = self.config.get("input_size",(240,240))
        self.categorys = self.config.get("categorys",["train","val"])
        self.main_path = self.config.get("main_path",os.path.join("pascal","VOCdevkit","VOC2012"))
        self.ignore_label = self.config.get("ignore_label",255)
        self.img_mean = np.ones((self.w,self.h,3))
        self.img_mean[:,:,0] *= 104.00698793
        self.img_mean[:,:,1] *= 116.66876762
        self.img_mean[:,:,2] *= 122.67891434
        self.init()

    def init(self):
        self.data_f,self.data_len = self.get_data_f()
        self.data = self.get_data()
        self.info = self.reset_info()

    def get_data_f(self):
        data_f = {}
        data_len = {}
        for category in self.categorys:
            data_f[category] = {"img":[],"label":[],"id":[]}
            data_len[category] = 0
        for one in self.categorys:
            with open(os.path.join("pascal","txt","%s.txt" % one),"r") as f:
                for line in f.readlines():
                    line = line.strip("\n") # the line is like "2007_000738"
                    data_f[one]["id"].append(line)
                    data_f[one]["img"].append(os.path.join(self.main_path,"JPEGImages","%s.jpg" % line))
                    data_f[one]["label"].append(os.path.join(self.main_path,"SegmentationClassAug","%s.png" % line))
            data_len[one] = len(data_f[one]["label"])

        print("len:%s" % str(data_len))
        return data_f,data_len

    def get_data(self):
        data = {}
        for category in self.categorys:
            data[category] = {"img":[],"label":[],"filename":[]}
        return data

    def reset_info(self):
        info = {}
        info["epoch"] = {} 
        info["index"] = {}
        info["perm"] = {}
        for category in self.categorys:
            info["epoch"][category] = 0
            info["index"][category] = 0
            perm = np.arange(self.data_len[category])
            np.random.shuffle(perm)
            info["perm"][category] = perm
        return info

    def get_info(self,key="epoch",category="train"):
        return self.info[key][category]

    def get_data_len(self,category="train"):
        return self.data_len[category]

    def get_cur_epoch(self,category="train"):
        return self.info["epoch"][category]


class dataset_np(dataset): 
    def __init__(self,config={}):
        super(dataset_np,self).__init__(config)

    def get_data(self):
        data = super(dataset_np,self).get_data() # create a empty data dict
        for category in self.categorys:
            for index in range(self.data_len[category]):
                img_f = self.data_f[category]["img"][index]
                label_f = self.data_f[category]["label"][index]
                img = imgio.imread(img_f)
                img = 255* skimage.img_as_float(img)
                label = imgio.imread(label_f)
                label = skimage.img_as_ubyte(label)
                img,label = self.preprocess(img,label)
                data[category]["img"].append(img)
                data[category]["label"].append(label)
                data[category]["filename"].append(os.path.basename(img_f)[:-4])
        return data

    def next_batch(self,batch_size=10,category="train",left_remove=True):
        if batch_size is None:
            batch_size = self.config.get("batch_size",1)
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

        ids = [self.data[category]["id"][index] for index in perm]
        imgs = [self.data[category]["img"][index] for index in perm]
        labels = [self.data[category]["label"][index] for index in perm]
        return np.array(imgs),np.array(labels),np.array(ids)

    def preprocess(self,img,label):
        h,w = img.shape[0:2]
        label -= self.ignore_label
        if w <= self.w:
            pad_width_1 = int(math.ceil((self.w - w)/2))
            pad_width_2 = int(math.floor((self.w - w)/2))
            img = np.pad(img,((0,0),(pad_width_1,pad_width_2),(0,0)),mode="constant",constant_values=0)
            label = np.pad(label,((0,0),(pad_width_1,pad_width_2)),mode="constant",constant_values=0)
            w_position = 0
        else:
            w_position = int(math.ceil((w-self.w)/2))
        if h <= self.h:
            pad_width_1 = int(math.ceil((self.h - h)/2))
            pad_width_2 = int(math.floor((self.h - h)/2))
            img = np.pad(img,((pad_width_1,pad_width_2),(0,0),(0,0)),mode="constant",constant_values=0)
            label = np.pad(label,((pad_width_1,pad_width_2),(0,0)),mode="constant",constant_values=0)
            h_position = 0
        else:
            h_position = int(math.ceil((h-self.h)/2))
        label += self.ignore_label
        img = img[h_position:h_position+self.h,w_position:w_position+self.w,:]
        label = label[h_position:h_position+self.h,w_position:w_position+self.w]

        ##RGB -> BGR
        r,g,b = np.dsplit(img,3)
        img = np.concatenate((b,g,r),axis=2)
        img -= self.img_mean
        # todo : random flip and random rotate

        return img,label

class dataset_tf(dataset):
    def __init__(self,config={},init=False):
        super(dataset_tf,self).__init__(config)

    def init(self):
        self.data_f,self.data_len = self.get_data_f()
        self.info = self.reset_info()

    def next_batch(self,category="train",batch_size=None,epoches=-1):
        if batch_size is None:
            batch_size = self.config.get("batch_size",1)
        dataset = tf.data.Dataset.from_tensor_slices({
            "id":self.data_f[category]["id"],
            "img_f":self.data_f[category]["img"],
            "label_f":self.data_f[category]["label"]
            })
        def m(x):
            id = x["id"]
            img_f = x["img_f"]
            img_raw = tf.read_file(img_f)
            img = tf.image.decode_image(img_raw)
            img = tf.expand_dims(img,axis=0)
            label_f = x["label_f"]
            label_raw = tf.read_file(label_f)
            label = tf.image.decode_image(label_raw)
            label = tf.expand_dims(label,axis=0)
            if category == "train":
                img,label = self.image_preprocess(img,label,random_scale=True,flip=True,rotate=True)
            else:
                img,label = self.image_preprocess(img,label,random_scale=False)

            img = tf.reshape(img,[self.h,self.w,3])
            label = tf.reshape(label,[self.h,self.w,1])

            return img,label,id

        dataset = dataset.repeat(epoches)
        dataset = dataset.shuffle(self.data_len[category])
        dataset = dataset.map(m)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        img,label,filename = iterator.get_next()
            
        return img,label,filename,iterator

    def image_preprocess(self,img,label,random_scale=True,flip=False,rotate=False):
        # input img and label shape [None, h, w, c]
        label -= self.ignore_label
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
        label += self.ignore_label
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

    def image_flip(self,img,label,left_right=True, up_down=False, random_s=0.5):
        if left_right is True:
            r = tf.random_uniform([1])
            r = tf.reduce_sum(r)
            img = tf.cond(r < random_s, lambda:tf.image.flip_left_right(img),lambda:img)
            label = tf.cond(r < random_s, lambda:tf.image.flip_left_right(label),lambda:label)
        if up_down is True:
            r = tf.random_uniform([1])
            r = tf.reduce_sum(r)
            img = tf.cond(r < random_s, lambda:tf.image.flip_left_right(img),lambda:img)
            label = tf.cond(r < random_s, lambda:tf.image.flip_left_right(label),lambda:label)

        return img, label

    def image_rotate(self,img,label,minangle=0,maxangle=0.314):
        angle = tf.random_uniform([1],minval=minangle,maxval=maxangle)
        angle = tf.squeeze(angle,[0])
        img = tf.contrib.image.rotate(img, angle)
        label = tf.contrib.image.rotate(label,angle)
        return img, label

def test_dataset_np():
    data= dataset_np({"input_size":(240,240),"categorys":["val"]})
    cur_epoch = data.get_cur_epoch(category="val")
    i = 0
    while cur_epoch == data.get_cur_epoch(category="val"):
        print("%d" % i)
        print("next_batch...")
        x,y = data.next_batch(category="val",batch_size=1)
        #x = np.squeeze(x/255,axis=0)
        #imgio.imsave("tmp/%d.png" % i,x)
        #imgio.imsave("tmp/%d_add_mean.png" % i, x+data.img_mean)
        i+=1

def test_dataset_tf():
    data= dataset_tf({"input_size":(240,240),"categorys":["val"]})
    x,y,f,iterator = data.next_batch(category="val",batch_size=4,epoches=1)
    sess = tf.Session()
    sess.run(iterator.initializer)
    i = 0
    try:
        while(True):
            if i % 10 == 0: print("i:%d" % i)
            _ = sess.run([x,y,f])
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
