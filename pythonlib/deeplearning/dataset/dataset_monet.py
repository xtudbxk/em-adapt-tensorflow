import numpy as np
import os
import copy
import random
import time
import tensorflow as tf
import sys
import math
from scipy import misc
from skimage import segmentation as sg
import pickle
from dataset import dataset_tf

class dataset_monet(dataset_tf):
    def __init__(self,config={}):
        self.use_split = self.config.get("use_split",True)
        self.split_num = self.config.get("split_num",8)
        self.preserve_num = self.config.get("preserve_num",1)
        super(dataset_monet,self).__init__(config,init=False)
        self.data_f,self.data_len = self.get_data_f()
        self.data = self.get_data_tf(self.config.get("batch_size",8))

    def get_data_f(self):
        data_f = {"train":{"x":[],"y":[]},"test":{"x":[],"y":[]},"val":{"x":[],"y":[]}}
        data_len = {"train":0,"test":0,"val":0}
        categorys = self.restore_or_save_categorys()
        if len(categorys) < 1:
            categorys_flag = False
        else:
            categorys_flag = True
        for one in ["train","val"]:
            with open(os.path.join("pascal_voc","txt","%s.txt" % one),"r") as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if random.random() < 2:
                        data_f[one]["x"].append(os.path.join("pascal_voc","images","%s.jpg" % line))
                        data_f[one]["y"].append(os.path.join("pascal_voc","annotations","%s.png" % line))
                        if not categorys_flag:
                            #print("%s"% line)
                            all_categorys = self.get_img_categorys_(data_f[one]["y"][-1])
                            for category in all_categorys:
                                #print("categorys:%s" % repr(categorys))
                                if category not in categorys:
                                    categorys[category] = set([data_f[one]["x"][-1]])
                                else:
                                    categorys[category].add(data_f[one]["x"][-1])
            #data_f[one]["x"] = data_f[one]["x"][:100]
            #data_f[one]["y"] = data_f[one]["y"][:100]
            data_len[one] = len(data_f[one]["y"])

        print("len:%s" % str(data_len))
        self.categorys = categorys
        categorys["all"] = set([])
        for category in categorys:
            categorys["all"].update(categorys[category])
        self.restore_or_save_categorys(categorys)
        return data_f,data_len

    def restore_or_save_categorys(self,categorys=None):
        path = "pascal_voc/categorys"
        if categorys is None:
            categorys = {}
            if os.path.exists(path):
                f = open(path,"rb")
                categorys = pickle.load(f)
                f.close()

            return categorys
        else:
            f = open(path,"wb")
            pickle.dump(categorys,f)
            f.close()

    def get_img_categorys_(self,label_filename):
        all_categorys = []
        l = misc.imread(label_filename)
        base_mask = np.ones(l.shape)
        for i in range(1,21):
            mask = i * base_mask
            equal = np.equal(mask,l)
            equal_n = equal.astype("float16")
            equal_ns = np.sum(equal_n)
            if equal_ns > 0:
                all_categorys.append(str(i))
        return all_categorys


    def get_img_categorys(self,img_filename):
        all_category = []
        for category in self.categorys:
            #print("img filename:%s" % img_filename)
            #print("category:%s %s" % (category,self.categorys[category]))
            if category == "all": continue
            if img_filename in self.categorys[category]:
            #    print("herre")
                all_category.append(category)
        #assert len(all_category)>0, "img has no categorys"
        return all_category



    def convert_img_without_split(self,img_filename,label_filename):

        img = misc.imread(img_filename)
        img = misc.imresize(img,(self.w,self.h))

        return img.astype(np.float32)

    def convert_img_with_split(self,img_filename,label_filename):
        #img = misc.imread(img_filename)
        #img = misc.imresize(img,(self.w,self.h))
        img_filename = img_filename.decode()
        label_filename = label_filename.decode()
        #return img
        filenames = []
        for i in range(self.preserve_num):
            filenames.append(img_filename)
        categorys = self.get_img_categorys(img_filename)
        all_filenames = copy.deepcopy(self.categorys["all"])
        for category in categorys:
            all_filenames -= self.categorys[category]
        #print("img name:%s" % img_filename)
        #print("categorys:%s" % str(categorys))
        for i in range(self.split_num-1):
            one_filename = all_filenames.pop()
            filenames.append(one_filename)
        #print("filenames:%s,%s" % (img_filename,str(filenames)))

        l_img = misc.imread(label_filename)
        l_img = misc.imresize(l_img,(self.w,self.h))
        l_tmp = np.greater(l_img,0)
        l_mask = l_tmp.astype("uint8")
        img = misc.imread(img_filename)
        img = misc.imresize(img,(self.w,self.h))

        masks = self.random_split(img,l_mask,self.w,self.h,num=self.split_num)
        img_z = np.zeros((self.w,self.h,3))
        for i,fname in enumerate(filenames[self.preserve_num:]):
            img_tmp = misc.imread(fname)
            img_tmp = misc.imresize(img_tmp,(self.w,self.h))
            img_z += img_tmp * masks[i+self.preserve_num]

        #misc.toimage(img_z,cmin=0,cmax=255).save("test/%s_masks_pre.jpg" % img_filename.decode()[:-4])
        img_z += img * masks[0]

        #misc.toimage(img_z,cmin=0,cmax=255).save("test/%s_masks.jpg" % img_filename.decode()[:-4])
        #misc.toimage(l_img*10,cmin=0,cmax=255).save("test/%s_label.jpg" % img_filename.decode()[:-4])

        #misc.toimage(img,cmin=0,cmax=255).save("test/%s" % img_filename.decode())
        return img_z.astype(np.float32)



    def random_split(self,img,label,w=224,h=224,num=8):
        s_num = 100
        img_s = sg.slic(img,s_num,1/10**random.randint(-3,3))
        all_s_num_dict = {}
        mask_s = np.greater(label,0) # the labeled mask
        #print("start:%f" % time.clock())
        for i in range(int(w/2)):
            for j in range(int(h/2)):
                key = str(img_s[2*i][2*j])
                if key not in all_s_num_dict:
                    all_s_num_dict[key] = 0
                if label[2*i][2*j] > 0:
                    all_s_num_dict[key] += 1
                    
        #print("all_s_num_dict:%s" % str(all_s_num_dict))
        all_s_num = [int(key) for key in all_s_num_dict if all_s_num_dict[key] <= 2]
        if len(all_s_num) <= 0:
            base_mask = np.ones((w,h,1))
            masks = [np.zeros((w,h,1)) for one in range(num)]
            masks.insert(0,base_mask)
            return masks

        while len(all_s_num) < num:
            all_s_num.extend(all_s_num)
        #print("midium:%f" % time.clock())

        np.random.shuffle(all_s_num)
        masks = []
        base_mask = np.ones((w,h,1))
        #print("all_s_num:%s" % str(all_s_num[:num]))
        #for i in range(30,40):
        #    print(str(img_s[i][:]))
        for one in all_s_num[:num]:
            ones_m = np.ones((w,h))*one
            mask = np.equal(ones_m,img_s).astype("uint8")
            #print("mask:%d" % np.sum(mask))
            mask = mask.reshape([self.w,self.h,1])
            base_mask -= mask
            masks.append(mask)
        masks.insert(0,base_mask)
        #print("a")
        #print("finally:%f" % time.clock())
        return masks
            



    def random_split_(self,label_mask,w=224,h=224,num=8):
        mask_probe = self.create_mask_probe(label_mask)
        #misc.imsave("test/pascal_voc/images/probe.jpg",mask_probe)
        masks = []
        base_mask = np.ones((self.w,self.h))
        for i in range(num-1):
            x = random.randint(0,w-1)
            y = random.randint(0,h-1)
            while mask_probe[x][y] < 0.1:
                x = random.randint(0,w-1)
                y = random.randint(0,h-1)
                #print("4")

            mask = self.random_split_mask(mask_probe,x,y,w,h,num)
            base_mask -= mask
            masks.append(mask)
            masks[i] = masks[i].reshape([self.w,self.h,1])
        masks.insert(0,base_mask)
        masks[0] = masks[0].reshape([self.w,self.h,1])

        return masks

    def create_mask_probe(self,mask):
        base = np.ones((self.w,self.h))
        d1 = 5
        d2 = 20
        for i in range(self.w):
            for j in range(self.h):
                if mask[i][j] <= 0:
                    if base[i][j] < 1.001:
                        base[i][j] = 0
                    else:
                        base[i][j] -= 1
                        if base[i][j] > 1:
                            base[i][j] = 1
                else:
                    for m,n in [(0,-1),(0,+1),(1,0),(-1,0)]:
                    #for m,n in [(0,+1),(1,0)]:
                        #print("here1")
                        if 0 < i+m < self.w and 0 < j+n < self.h:
                            #print("here3")
                            if mask[i+m][j+n] <= 0.1:
                                #print("here2")
                                for k in range(d2):
                                    if 0 < i+m*k < self.w and 0 < j+n*k < self.h:
                                        #print("here5")
                                        if k < d1:
                                            #print("here6")
                                            if m < 0 or n < 0:
                                                base[i+m*k][j+n*k] = 1
                                            else:
                                                base[i+m*k][j+n*k] += 1
                                                if base[i+m*k][j+n*k] > 2:
                                                    base[i+m*k][j+n*k] = 2

                                        else:
                                            #print("here7")
                                            if m < 0 or n < 0:
                                                base[i+m*k][j+n*k] += 1 - k / d2
                                                if base[i+m*k][j+n*k] > 1:
                                                    base[i+m*k][j+n*k] = 1
                                            else:
                                                base[i+m*k][j+n*k] += 1 - k / d2
                                                if base[i+m*k][j+n*k] > 2:
                                                    base[i+m*k][j+n*k] = 2
        for i in range(self.w):
            for j in range(self.h):
                if base[i][j] > 1:
                    base[i][j] = 1

        base = np.ones((self.w,self.h)) - base
        return base


    def random_split_mask(self,mask_probe,x,y,w,h,num=8):
        active_positions = set([(x,y)])
        mask = 0*mask_probe
        mask[x][y] = 1
        num_1 = 1
        M = int(h*w/num) * 3
        p_st = 1 
        while len(active_positions) != 0:
            #print("3")
            pos_x,pos_y = active_positions.pop() 
            if mask_probe[pos_x][pos_y] < 1e-2: continue
            for row,col in [(0,1),(0,-1),(1,0),(-1,0)]:
                if not ( 0 <= pos_x + col < w and 0 <= pos_y + row < h):
                    continue
                a = random.random()
                if a < p_st*mask_probe[pos_x][pos_y]:
                    active_positions.add((pos_x+col,pos_y+row))
                    mask[pos_x+col][pos_y+row] = 1
                    num_1 += 1
                    if num_1 > 4*M/5:
                        p_st -= p_st*3/M/2

        return mask

def test():
    h = 5
    w = 2
    sess = tf.Session()
    c = dataset_pascal()
    x = tf.placeholder(shape=[h,w,3],dtype=tf.float16)
    y = tf.placeholder(shape=[h,w,1],dtype=tf.float16)
    m,n = c.image_preprocess(x,y)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    a = np.ones([h,w,3])
    b = np.ones([h,w,1])
    h_old,w_old = sess.run([c.h_old,c.w_old],feed_dict={x:a,y:b})
    print("h_old:%d,w_old:%d" % (h_old,w_old))
    s,t = sess.run([m,n],feed_dict={x:a,y:b})
    print("s:%s" % str(s.shape))
    print("t:%s" % str(t.shape))
    print("s:%s" % str(s))
    print("t:%s" % str(t))

    

if __name__ == "__main__":
    #test()
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    c = dataset_monet({"use_split":True})
    x,y,_ = c.next_data_tf()
    #with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = time.time()
    for i in range(100):
        print("i:%d" % i)
        #print("queue:%s" % sess.run(c.queues[0]))
        print("data:%s" % sess.run(c.data["train"]["x"])[0][100][100])
        #print("names:%s" % sess.run(c.data["train"]["name"])[0])
        #print("data:%s" % sess.run(c.data["train"]["y"]))
        #a,b = sess.run([x,y])
        #print("a:%s,b:%s" % (a,b))
    end_time = time.time()
    print("total time:%f" % (end_time - start_time))
    coord.request_stop()
    coord.join(threads)
