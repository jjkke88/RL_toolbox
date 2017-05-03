#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
import os
import sys
import numpy as np
import time
from scipy.misc import imread, imresize
import os
import random
import cv2
from RL_classify.pano_sence_analysis.enviroment import Enviroment

"""
VGG-16 for ImageNet
Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper “Very Deep Convolutional Networks for
Large-Scale Image Recognition”  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.
Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow
Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping, as shown in the
following snippet:
>>> image_h, image_w, _ = np.shape(img)
>>> shorter_side = min(image_h, image_w)
>>> scale = 224. / shorter_side
>>> image_h, image_w = np.ceil([scale * image_h, scale * image_w]).astype('int32')
>>> img = imresize(img, (image_h, image_w))
>>> crop_x = (image_w - 224) / 2
>>> crop_y = (image_h - 224) / 2
>>> img = img[crop_y:crop_y+224,crop_x:crop_x+224,:]
"""

good_weld_img_folds = ["/media/disk/wyp/V6C/white/Test/OK"]
bad_weld_img_folds = ["/media/disk/wyp/V6C/white/Test/NG"]


def generate_labeled_filelist(path, label):
    file_list = os.listdir(path)

    labeled_list = []
    if file_list:
        for fn in file_list:
            full_file_name = os.path.join(path, fn)
            labeled_list.append([full_file_name, label])
    return labeled_list

def generate_filelists():
    file_lists = []
    for path in good_weld_img_folds:
        file_lists = file_lists + generate_labeled_filelist(path, 1)
    for path in bad_weld_img_folds:
        file_lists = file_lists + generate_labeled_filelist(path, 0)
    return file_lists

def choose_files(lists,train_percent, val_percent, test_percent, max_number = 0):
    random.shuffle(lists)
    print len(lists),"files wait to be load ... "
    if max_number != 0 and len(lists) > max_number:
        lists = lists[0:max_number]

    max_len = len(lists)
    train_number = int(max_len * train_percent)
    val_number = int(max_len * val_percent)
    test_number = int(max_len * test_percent)

    train_lists = lists[0:train_number]
    val_lists = lists[train_number:train_number+val_number]
    test_lists = lists[train_number+val_number:train_number+val_number+test_number]

    return train_lists, val_lists, test_lists

def load_dataset():
    from RL_classify.single_step.parameters import PMS_base as pms
    train_env = Enviroment(pms.train_file)
    test_env = Enviroment(pms.test_file)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    # for i in xrange(30000):
    #     view, label = train_env.generate_new_scence()
    #     X_train.append(view)
    #     y_train.append(label)
    # for i in xrange(3000):
    #     view , label = train_env.generate_new_scence()
    #     X_val.append(view)
    #     y_val.append(label)
    # for i in xrange(1000):
    #     view , label = test_env.generate_new_scence()
    #     X_test.append(view)
    #     y_test.append(label)
    from annotation import annotation , seperate_train_val_data
    from keras.preprocessing import image as keras_image
    with_bbox = False
    color_mode = 'rgb'  # or gray
    feature_center = True  # or False
    horizontal_flip = True  # or False
    vertical_flip = True  # or False
    train_val_ratio = 0.8
    d = annotation(
        image_set="train" ,
        data_path="/home/aqrose/RL_toolbox/RL_classify/single_step/dataset/" ,
        with_bbox=with_bbox)
    x , y = d.prepare_keras_data(target_size=224 , color_mode=color_mode)
    # y = np.concatenate([[np.argmax(y_)] for y_ in y])
    (x_train , y_train) , (x_test , y_test) = seperate_train_val_data(x , y , ratio=train_val_ratio)
    y_train = np.concatenate([[np.argmax(y_)] for y_ in y_train])
    y_test = np.concatenate([[np.argmax(y_)] for y_ in y_test])
    x_val = x_test
    y_val = y_test
    x_test = x_val[:100, :, :, :]
    y_test = y_val[:100]
    return x_train, y_train, x_test, y_test, x_test, y_test

def conv_layers(net_in):
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
    """ conv1 """
    network = tl.layers.Conv2dLayer(net_in,
                    act = tf.nn.relu,
                    shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool3')
    """ conv4 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool4')
    """ conv5 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_3')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool5')
    return network

def fc_layers(net):
    network = tl.layers.FlattenLayer(net, name='flatten')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu,
                        name = 'fc1_relu')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu,
                        name = 'fc2_relu')
    network = tl.layers.DenseLayer(network, n_units=26,
                        act = tf.identity,
                        name = 'fc3_relu')
    return network

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

net_in = tl.layers.InputLayer(x, name='input_layer')
net_cnn = conv_layers(net_in)
network = fc_layers(net_cnn)

y = network.outputs
probs = tf.nn.softmax(y)
# y is network result, and y_ is real result, y_op is the classify result
y_op = tf.argmax(tf.nn.softmax(y), 1)
cost = tl.cost.cross_entropy(y, y_, name='cost')
# optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001 , beta1=0.9 , beta2=0.999 ,
                                  epsilon=1e-08 , use_locking=False).minimize(cost , var_list=train_params)
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# sess.run(tf.initialize_all_variables())
tl.layers.initialize_global_variables(sess)
# network.print_params()
# network.print_layers()

## load pretrain model
# if not os.path.isfile("model.npz"):
#     print("Please download vgg16_weights.npz from : http://www.cs.toronto.edu/~frossard/post/vgg16/")
#     exit()
# npz = np.load('model.npz')
# params = []
# for val in sorted( npz.items() ):
#     print("  Loading %s" % str(val[1].shape))
#     params.append(val[1])
# tl.files.assign_params(sess, params, network)

# npz = np.load('vgg16_weights.npz')
# params = []
# for val in sorted( npz.items() ):
#     print("  Loading %s" % str(val[1].shape))
#     params.append(val[1])
# tl.files.assign_params(sess, npz, network)

## load data
X_train , y_train , X_val , y_val , X_test , y_test = load_dataset()

# train
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=32, n_epoch=100      , print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

tl.files.load_and_assign_npz(sess, 'model.npz', network)

# evaluate
from RL_classify.pano_sence_analysis.enviroment import Enviroment
from RL_classify.single_step.parameters import PMS_base as pms
test_env = Enviroment(pms.test_file)
all_view_container = []
all_label_containr = []
for i in xrange(500):
    view, label = test_env.generate_new_scence()
    all_view_container.append(view)
    all_label_containr.append(label)
X_test = np.array(all_view_container)
y_test = np.array(all_label_containr)
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=32, cost=cost)

# save model
# tl.files.save_npz(network.all_params , name='model.npz', sess=sess)

## for test
# img1 = imread('/media/disk/wyp/V6C/white/Test/OK/0070.bmp', mode='RGB') # test data in github
# img1 = imresize(img1, (224, 224))
#
# start_time = time.time()
# prob, predict_y = sess.run([probs, y_op], feed_dict={x: X_train})
# print predict_y
# "==========================="
# print y_train
# print("  End time : %.5ss" % (time.time() - start_time))
# prob = prob[0]
# predict_y = predict_y[0]
# preds = (np.argsort(prob)[::-1])[0:5]
# print predict_y
# for p in preds:
#     print(class_names[p], prob[p])
    # print p

sess.close()