import tensorflow as tf
import numpy as np
import skimage as sk
import skimage.io as imgio

def metrics_tf(gt,pred,num,shape,kinds=["miou"]):
    ret = {"t_i":[],"n_j_i":[],"n_i_i":[]}
    for i in range(num):
        ret["t_i"].append(tf.Variable(0.0,trainable=False))
        ret["n_j_i"].append(tf.Variable(0.0,trainable=False))
        ret["n_i_i"].append(tf.Variable(0.0,trainable=False))

    update_ops = []
    reset_ops = []
    for i in range(num): # ignore void label
        tmp = np.full(shape,i)
        t_i = tf.equal(gt,tmp)
        n_j_i = tf.equal(pred,tmp)
        n_i_i = tf.logical_and(t_i, n_j_i)
        #ret["t_i"].append(tf.reduce_sum(tf.cast(t_i,"float")))
        update_ops.append( tf.assign( ret["t_i"][i], ret["t_i"][i]+tf.reduce_sum(tf.cast(t_i,"float"))))
        #ret["n_j_i"].append(tf.reduce_sum(tf.cast(n_j_i,"float")))
        update_ops.append( tf.assign( ret["n_j_i"][i], ret["n_j_i"][i]+tf.reduce_sum(tf.cast(n_j_i,"float"))))
        #ret["n_i_i"].append(tf.reduce_sum(tf.cast(n_i_i,"float")))
        if i == num -1:
            with tf.control_dependencies(update_ops):
                update_op = tf.assign( ret["n_i_i"][i], ret["n_i_i"][i] + tf.reduce_sum(tf.cast(n_i_i,"float")))
        else:
            update_ops.append( tf.assign( ret["n_i_i"][i],ret["n_i_i"][i] + tf.reduce_sum(tf.cast(n_i_i,"float"))))

        reset_ops.append( tf.assign( ret["t_i"][i], 0.0))
        reset_ops.append( tf.assign( ret["n_j_i"][i], 0.0))
        reset_ops.append( tf.assign( ret["n_i_i"][i], 0.0))
        if i == num -1:
            with tf.control_dependencies(reset_ops):
                reset_op = tf.assign( ret["n_i_i"][i], 0.0)
        else:
            reset_ops.append( tf.assign( ret["n_i_i"][i], 0.0))

    num_of_existing_classes = 0
    metrics = []
    if "accuracy" in kinds:
        for i in range(num):
            num_of_existing_classes += tf.cond(ret["t_i"][i] > 0, lambda: 1.0, lambda: 0.0)
        # pixel accuracy
        pixel_accu = sum(ret["n_i_i"]) / sum(ret["t_i"])
        metrics.append(pixel_accu)

    # mean accuracy
    #mean_pixel_accu = sum([ ret["n_i_i"][i] / ret["t_i"][i] for i in range(num) if ret["t_i"][i] > 0]) / num_of_existing_classes
    #metrics.append(mean_pixel_accu)

    # IoU
    #IoUs = [ ret["n_i_i"][i] /  (ret["t_i"][i] + ret["n_j_i"][i] - ret["n_i_i"][i]) for i in range(num) if ret["t_i"][i] > 0]
    #metrics.append(IoUs)

    # mean IoU
    if "miou" in kinds:
        mIoU = 0
        for i in range(num):
            mIoU += tf.cond(ret["t_i"][i] > 0, lambda: ret["n_i_i"][i] / (ret["t_i"][i]+ret["n_j_i"][i]-ret["n_i_i"][i]), lambda: 0.0)
        mIoU /= num_of_existing_classes
        metrics.append(mIoU)

    # frequency weighted IU
    if "fiou" in kinds:
        fIoU = 0
        for i in range(num):
            fIoU += tf.cond(ret["t_i"][i] > 0, lambda: ret["t_i"][i] * ret["n_i_i"][i] / (ret["t_i"][i]+ret["n_j_i"][i]-ret["n_i_i"][i]), lambda: 0.0)
        fIoU /= sum(ret["t_i"])
        metrics.append(fIoU)

    metrics.append(update_op)
    metrics.append(reset_op)
    return metrics

def metrics_update(loss,optimizer,kinds=["gradient","weight","rate"],first_n=-1,summarize=3):
    gradients = optimizer.compute_gradients(loss)
    gradients = [(g,v) for (g,v) in gradients if g is not None]
    a = tf.Variable(1.0,dtype=tf.float32)
    for (g,v) in gradients:
        if "gradient" in kinds:
            a = tf.Print(a,[g.name,tf.reduce_mean(g)],"gradient",first_n=first_n,summarize=summarize)
        if "weight" in kinds:
            a = tf.Print(a,[v.name,tf.reduce_mean(v)],"weight",first_n=first_n,summarize=summarize)
        if "rate" in kinds:
            a = tf.Print(a,[v.name,tf.reduce_mean(g)/(tf.reduce_mean(v)+1e-20)],"rate",first_n=first_n,summarize=summarize)
    optim = optimizer.apply_gradients(gradients)
    return optim,a


# Originally written by wkentaro for the numpy version
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
class metrics_np():
    def __init__(self,n_class=1,hist=None):
        if hist is None:
            self.hist = np.zeros((n_class,n_class))
        else:
            self.hist = hist
        self.n_class = n_class

    def _fast_hist(self,label_true,label_pred,n_class):
        mask = (label_true>=0) & (label_true<n_class) # to ignore void label
        self.hist = np.bincount( n_class * label_true[mask].astype(int)+label_pred[mask],minlength=n_class**2).reshape(n_class,n_class)
        return self.hist

    def update(self,x,y):
        self.hist += self._fast_hist(x.flatten(),y.flatten(),self.n_class)

    def get(self,kind="miou"):
        if kind == "accu":
            return np.diag(self.hist).sum() / self.hist.sum() # total pixel accuracy
        elif kind == "accus":
            return np.diag(self.hist) / self.hist.sum(axis=1) # pixel accuracys for each category, np.nan represent the corresponding category not exists
        elif kind in ["freq","fiou","iou","miou"]:
            iou = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0) - np.diag(self.hist))
            if kind == "iou": return iou
            miou = np.nanmean(iou)
            if kind == "miou": return miou

            freq = self.hist.sum(axis=1) / self.hist.sum() # the frequency for each categorys
            if kind == "freq": return freq
            else: return (freq[freq>0]*iou[freq>0]).sum()
        elif kind in ["dice","mdice"]:
            dice = 2*np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0))
            if kind == "dice": return dice
            else: return np.nanmean(dice)
        return None

    def get_all(self):
     metrics = {}
     metrics["accu"] = np.diag(self.hist).sum() / self.hist.sum() # total pixel accuracy
     metrics["accus"] = np.diag(self.hist) / self.hist.sum(axis=1) # pixel accuracys for each category, np.nan represent the corresponding category not exists
     metrics["iou"] = np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0) - np.diag(self.hist))
     metrics["miou"] = np.nanmean(metrics["iou"])
     metrics["freq"] = self.hist.sum(axis=1) / self.hist.sum() # the frequency for each categorys
     metrics["fiou"] = (metrics["freq"][metrics["freq"]>0]*metrics["iou"][metrics["freq"]>0]).sum()
     metrics["dices"] = 2*np.diag(self.hist) / (self.hist.sum(axis=1)+self.hist.sum(axis=0))
     metrics["mdice"] = np.nanmean(metrics["dices"])
 
     return metrics

def test_np():
    gt = np.array([[1,0],[1,1]],dtype=np.uint8)
    pred = np.array([[1,0],[1,0]],dtype=np.uint8)
    m = metrics_np(n_class=2)
    m.update(gt,pred)
    print("iou:%s" % str(m.get("iou")))
    print("all:%s" % str(m.get_all()))

if __name__ == "__main__":
    test_np()
