import tensorflow as tf
from metrics import metrics_tf
from dataset import dataset_tf as dataset
import numpy as np
import time

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
        self.metrics = {}
        self.summary = {"train":{"op":None},"val":{"op":None}}
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
        return layer

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
    
    # need to rewrite
    def getloss(self):
        label,output = self.remove_ignore_label(self.net["label"],self.net["output"])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=output))
        return loss

    # need to rewrite
    def optimize(self,base_lr,momentum):
        self.net["lr"] = tf.Variable(base_lr, trainable=False)
        opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        gradients = opt.compute_gradients(self.loss["total"],var_list=self.trainable_list)
        a = tf.Variable(1.0,dtype=tf.float32)
        for (g,v) in gradients:
            a = tf.Print(a,[v.name,tf.reduce_mean(tf.abs(g))],"gradientmean")
            a = tf.Print(a,[v.name,tf.reduce_max(g)],"gradientmax")
            a = tf.Print(a,[v.name,tf.reduce_min(g)],"gradientmin")
            a = tf.Print(a,[v.name,tf.reduce_mean(tf.abs(v))],"weightmean")
            a = tf.Print(a,[v.name,tf.reduce_max(v)],"weightmax")
            a = tf.Print(a,[v.name,tf.reduce_min(v)],"weightmin")
            b = g/(v+1e-20)
            a = tf.Print(a,[v.name,tf.reduce_mean(tf.abs(b))],"ratemean")
            a = tf.Print(a,[v.name,tf.reduce_max(b)],"ratemax")
            a = tf.Print(a,[v.name,tf.reduce_min(b)],"ratemin")
        self.net["g"] = a

        self.net["accum_gradient_accum"] = [self.net["accum_gradient"][i].assign_add( g[0]/self.accum_num ) for (i,g) in enumerate(gradients)]
        self.net["accum_gradient_clean"] = [g.assign(tf.zeros_like(g)) for g in self.net["accum_gradient"]]
        gradients = [(g,self.trainable_list[i]) for i,g in enumerate(self.net["accum_gradient"])]
        self.net["accum_gradient_update"]  = opt.apply_gradients(gradients)
        self.net["train_op"] = opt.apply_gradients(gradients)


    # need to rewrite
    def getmetrics(self,batch_size=10):
        self.metrics["accu"],self.metrics["miou"],self.metrics["update"],self.metrics["reset"] = metrics_tf(tf.squeeze(self.net["label"],axis=3),self.net["pred"],self.category_num,shape=[batch_size,self.h,self.w],kinds=["accuracy","miou"])
        return ["accu","miou"]

    # need to rewrite 86lines
    def train(self,base_lr,weight_decay,momentum,batch_size,epoches):
        assert self.data is not None,"data is None"
        assert self.sess is not None,"sess is None"
        self.net["is_training"] = tf.placeholder(tf.bool)

        x_train,y_train,id_train,iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        x_val,y_val,id_val,iterator_val = self.data.next_batch(category="val",batch_size=batch_size,epoches=-1)
        x = tf.cond(self.net["is_training"],lambda:x_train,lambda:x_val)
        y = tf.cond(self.net["is_training"],lambda:y_train,lambda:y_val)
        self.build()
        self.pre_train(base_lr,weight_decay,momentum,batch_size,save_layers=["input","output","label","pred","is_training"])

        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
        self.sess = tf.Session(config=gpu_options)
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            self.sess.run(iterator_val.initializer)

            if self.config.get("model_path",False) is not False:
                print("start to load model: %s" % self.config.get("model_path"))
                print("before l2 loss:%f" % self.sess.run(self.loss["l2"]))
                self.restore_from_model(self.saver["norm"],self.config.get("model_path"),checkpoint=False)
                print("model loaded ...")
                print("after l2 loss:%f" % self.sess.run(self.loss["l2"]))

            start_time = time.time()
            print("start_time: %f" % start_time)
            print("config -- lr:%f weight_decay:%f momentum:%f batch_size:%f epoches:%f" % (base_lr,weight_decay,momentum,batch_size,epoches))

            epoch,i = 0.0,0
            iterations_per_epoch_train = self.data.get_data_len() // batch_size
            while epoch < epoches:
                if i == 0: # to protect restore
                    self.sess.run(tf.assign(self.net["lr"],base_lr))
                if i == 40*iterations_per_epoch_train:
                    new_lr = 0.003
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"lr-%f" % base_lr),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))

                data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:True})
                params = {self.net["input"]:data_x,self.net["label"]:data_y}
                _ = self.sess.run(self.net["train_op"],feed_dict=params)

                if i%100 in [0,1,2,3,4,5,6,7,8,9]:
                    self.sess.run(self.metrics["update"],feed_dict=params)
                if i%100 == 9:
                    summarys,accu,miou,loss,lr = self.sess.run([self.summary["train"]["op"],self.metrics["accu"],self.metrics["miou"],self.loss["total"],self.net["lr"]],feed_dict=params)
                    self.summary["writer"].add_summary(summarys,i)
                    print("epoch:%f, iteration:%f, lr:%f, loss:%f, accu:%f, miou:%f" % (epoch,i,lr,loss,accu,miou))
                if i%100 == 10:
                    self.sess.run(self.metrics["reset"],feed_dict=params)

                if i%1000 in [10,11,12,13,14,15,16,17,18,19]:
                    data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:False})
                    params = {self.net["input"]:data_x,self.net["label"]:data_y}
                    self.sess.run(self.metrics["update"],feed_dict=params)
                if i%1000 == 19:
                    data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:False})
                    params = {self.net["input"]:data_x,self.net["label"]:data_y}
                    summarys,accu,miou,loss,lr = self.sess.run([self.summary["val"]["op"],self.metrics["accu"],self.metrics["miou"],self.loss["total"],self.net["lr"]],feed_dict=params)
                    self.summary["writer"].add_summary(summarys,i)
                    print("val epoch:%f, iteration:%f, lr:%f, loss:%f, accu:%f, miou:%f" % (epoch,i,lr,loss,accu,miou))
                if i%1000 == 20:
                    data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:False})
                    params = {self.net["input"]:data_x,self.net["label"]:data_y}
                    self.sess.run(self.metrics["reset"],feed_dict=params)

                if i%3000 == 2999:
                    self.saver["norm"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"norm"),global_step=i)
                i+=1
                epoch = i / iterations_per_epoch_train

            end_time = time.time()
            print("end_time:%f" % end_time)
            print("duration time:%f" %  (end_time-start_time))

    def loss_summary(self,weight_decay):
            self.loss["norm"] = self.getloss()
            self.loss["l2"] = sum([tf.nn.l2_loss(self.weights[layer][0]) for layer in self.weights])
            self.loss["total"] = self.loss["norm"] + weight_decay*self.loss["l2"]
            return ["total","l2","norm"]

    def image_summary(self):
            upsample = 250/self.category_num
            gt_single = tf.to_float(self.net["label"])*upsample
            gt_rgb = tf.concat([gt_single,gt_single,gt_single],axis=3)
            pred_single = tf.to_float(tf.reshape(self.net["pred"],[-1,self.h,self.w,1]))*upsample
            pred_rgb = tf.concat([pred_single,pred_single,pred_single],axis=3)
            self.images["image"] = tf.concat([tf.cast(self.net["input"]+self.data.img_mean,tf.uint8),tf.cast(gt_rgb,tf.uint8),tf.cast(pred_rgb,tf.uint8)],axis=2)
            return ["image"]

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
        assert self.sess is not None,"sess is None"
        self.net["accum_gradient"] = [tf.Variable(tf.zeros_like(v),trainable=False) for v in self.trainable_list]
        summarys = {}
        with self.sess.as_default():
            loss_summarys = self.loss_summary(weight_decay)
            for loss in loss_summarys:
                summarys[loss] = self.loss[loss]

            self.optimize(base_lr,momentum)

            metric_summarys = self.getmetrics(batch_size)
            for metric in metric_summarys:
                summarys[metric] = self.metrics[metric]

            image_summarys = self.image_summary()
            for image in image_summarys:
                summarys[image] = self.images[image]

            for category in ["train","val"]:
                for s in summarys:
                    if s == "image":
                        self.summary[category][s] = tf.summary.image("%s_%s"%(category,s),summarys[s],max_outputs=batch_size)
                    else:
                        self.summary[category][s] = tf.summary.scalar("%s_%s"%(category,s),summarys[s])
            for category in ["train","val"]:
                summary_list = []
                for s in self.summary[category]:
                    if s != "op":
                        summary_list.append(self.summary[category][s])
                self.summary[category]["op"] = tf.summary.merge(summary_list)
            self.summary["writer"] = tf.summary.FileWriter(self.config.get("log","log"),graph=self.sess.graph)

            for layer in save_layers:
                tf.add_to_collection(layer,self.net[layer])

            self.saver["norm"] = tf.train.Saver(max_to_keep=3,var_list=self.trainable_list)
            self.saver["lr"] = tf.train.Saver(var_list=self.trainable_list)

    def restore_from_model(self,saver,model_path,checkpoint=False):
        assert self.sess is not None
        if checkpoint is True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, model_path)

    def tensorboard(self):
        sess = tf.Session()
        tf.summary.scalar("output",tf.reduce_sum(self.net["output"]))
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/",sess.graph)
        sess.run(tf.global_variables_initializer())
        a = sess.run(merged,feed_dict={self.net["input"]:np.ones([1,self.config["input_h"],self.config["input_w"],3]),self.net["is_training"]:True})
        writer.add_summary(a)
        
if __name__ == "__main__":
    n = Network()
    n.tensorboard()
