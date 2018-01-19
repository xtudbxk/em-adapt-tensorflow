import tensorflow as tf
import numpy as np
import skimage as sk
import skimage.io as imgio


def metrics_summary():
    pixel_accu_ph = tf.placeholder(dtype=tf.float16)
    pixel_accu_s_train = tf.summary.scalar("train_pixel_accu_train",pixel_accu_ph)
    pixel_accu_s_test = tf.summary.scalar("pixel_accu_test",pixel_accu_ph)
    mIoU_ph = tf.placeholder(dtype=tf.float16)
    mIoU_s_train = tf.summary.scalar("mIoU_train",mIoU_ph)
    mIoU_s_test = tf.summary.scalar("mIoU_test",mIoU_ph)
    tf.summary.scalar("mIoU",mIoU_ph)
    fIoU_ph = tf.placeholder(dtype=tf.float16)
    fIoU_s_train = tf.summary.scalar("fIoU_train",fIoU_ph)
    fIoU_s_test = tf.summary.scalar("fIoU_test",fIoU_ph)

    train_s = tf.summary.merge([pixel_accu_s_train,mIoU_s_train,fIoU_s_train])
    test_s = tf.summary.merge([pixel_accu_s_test,mIoU_s_test,fIoU_s_test])
    return pixel_accu_ph,mIoU_ph,fIoU_ph,train_s,test_s

def accuracy(gt,pred,one_hot=False):
    accu_n = np.sum(np.equal(gt,pred).astype("int16"))
    return accu_n / gt.shape[0]

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
def _fast_hist(label_true,label_pred,n_class):
    mask = (label_true>=0) & (label_true<n_class) # to ignore void label
    hist = np.bincount( n_class * label_true[mask].astype(int)+label_pred[mask],minlength=n_class**2).reshape(n_class,n_class)
    return hist

def metrics_np(label_trues,label_preds,n_class): # note label_trues and label_preds are all predicting images and different images is concate by list type
    hist = np.zeros((n_class,n_class))
    for lt,lp in zip(label_trues,label_preds):
        hist += _fast_hist(lt.flatten(),lp.flatten(),n_class)
    metrics = {}
    metrics["accu"] = np.diag(hist).sum() / hist.sum() # total pixel accuracy
    metrics["accus"] = np.diag(hist) / hist.sum(axis=1) # pixel accuracys for each category, np.nan represent the corresponding category not exists
    metrics["iou"] = np.diag(hist) / (hist.sum(axis=1)+hist.sum(axis=0) - np.diag(hist))
    metrics["miou"] = np.nanmean(metrics["iou"])
    metrics["freq"] = hist.sum(axis=1) / hist.sum() # the frequency for each categorys
    metrics["fiou"] = (metrics["freq"][metrics["freq"]>0]*metrics["iou"][metrics["freq"]>0]).sum()
    metrics["dices"] = 2*np.diag(hist) / (hist.sum(axis=1)+hist.sum(axis=0))
    metrics["mdice"] = np.nanmean(metrics["dices"])

    return metrics

if __name__ == "__main__":
    f = "pascal_voc/annotations/2007_000032.png"
    gt = imgio.imread(f)
    gt = sk.img_as_ubyte(gt)
    pred = gt
    print("iou:%s" % str(metrics_np(gt,pred,21)))

