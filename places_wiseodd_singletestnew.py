#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
places_wiseodd.py

Created on Thu May 18 09:25:50 2017

@author: paul
"""

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import vgg
import time
from PIL import Image
import scipy.misc
from scipy.misc import imread
from plotloss import plotloss2y

def rgb2gray(rgb):
    if np.ndim(rgb)==3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        return rgb

def gray2rgb(gray):
    if np.ndim(gray)==2:
        w, h = gray.shape
        rgb = np.empty((w, h, 3), dtype=np.float32)
        rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
        return rgb
    else:
        return gray


L_data = 9
mb_size = 2 
Z_dim = 256                                    
X_dim = 256
h_dim = 128
coeff = 1e-7
num_it = 100000

sess = tf.Session()
 
print(1)
places = os.listdir('./abbey/')
data=[]
for i in range(L_data):
    #data.append(np.reshape(rgb2gray(imread('./abbey/' + places[i])),(1,X_dim)))
    data.append(gray2rgb(imread('./abbey/' + places[i]))/256)
    print('Image ',i+1,' : ',places[i])
    print('data[i].shape',data[i].shape)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)    #standard deviation of the normal distribution
    return tf.random_normal(shape=size, stddev=xavier_stddev)  #donne des valeurs aléatoires d'une distribution normale


def next_batch(batch_size,index_in_epoch):
    start = index_in_epoch
    if start + batch_size > L_data:
        np.random.shuffle(data)
        start = 0
        index_in_epoch = batch_size
    else:
        index_in_epoch += batch_size
    end = index_in_epoch 
    D = np.vstack(data[start:end])
    return D, index_in_epoch



print(2)
""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[mb_size, X_dim, X_dim, 3])
#y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim, X_dim, 3, h_dim]))
# X_dim + y_dim plutôt que X_dim comme ça on train non seulement à dire si l'image appartient ou non au dataset original, MAIS AUSSI dire à quel label elle correspond !
D_b1 = tf.Variable(tf.zeros(shape=[mb_size,h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[mb_size,1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]  # Parameters to optimize
# => initial : D_W = random & D_b = zeros ; then = optimize to get best discriminator
# D_W1 : shape=(794,128)  ;  D_W2 : shape=(128,1)  Weights pour passer d'un layer à l'autre  (input -->D_W1--> relu -->D_W2--> result)
# D_b1 : shape=(128,)  ;  D_b2 : shape=(1,)    Biases

print(3)
def discriminator(x):                                 # Single layer network 
    #inputs = tf.concat(axis=1, values=[x, y])
    inputs=x
    D_h1 = tf.nn.relu(tf.einsum('ijkl,jklm->im',inputs, D_W1) + D_b1)    # = max( 0 , inputs × D_W1 + D_b1 )      
    D_logit = tf.matmul(D_h1, D_W2) + D_b2               # include weights and biases
    D_prob = tf.nn.sigmoid(D_logit)                      # to have a result between 0 and 1 (probability)

    return D_prob, D_logit    # /!\ not scalar : operations /!\


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[mb_size, Z_dim, Z_dim, 3])

G_W1 = tf.Variable(xavier_init([Z_dim, Z_dim, 3, h_dim])) 
G_b1 = tf.Variable(tf.zeros(shape=[mb_size,h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim,X_dim,X_dim, 3]))
G_b2 = tf.Variable(tf.zeros(shape=[mb_size,X_dim,X_dim, 3]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


#G_W1 : shape=(110,128)  ;  G_W2 : shape=(128,784)
#G_b1 : shape=(128,)  ;  G_b2 : shape=(784,)
print(4)

def generator(z):
    inputs=z                                    # shape = (mb_size, Z_dim, Z_dim)
    G_h1 = tf.nn.relu(tf.einsum('ijkl,jklm->im', inputs, G_W1) + G_b1)
    G_log_prob = tf.einsum('ij,jklm->iklm',G_h1, G_W2) + G_b2   # shape = (mb_size, X_dim, X_dim)
    G_prob = tf.nn.sigmoid(G_log_prob)
    print("input:", inputs.shape, "G_log:", G_log_prob.get_shape().as_list(), "G_prob:", G_prob.get_shape().as_list())
    return G_prob


# def sample_Z(m, n):
    # return np.random.uniform(-1., 1., size=[m, n])
print('ok!!!')

def plot(sample,X):
    fig = plt.figure()
    gs = gridspec.GridSpec(mb_size,2)
    gs.update(wspace=0.05,hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[2*i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
        plt.imshow(gray2rgb(sample))
        ax = plt.subplot(gs[2*i+1])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.imshow(np.reshape(gray2rgb(X[i]),(256,256,3)))

    return fig

G_sample = generator(Z)    # Z = random input images to begin with.
D_real, D_logit_real = discriminator(X)   # X = real dataset
D_fake, D_logit_fake = discriminator(G_sample)  # G_sample = generated dataset (fake)
# D_real & D_fake = unused   (D_fake = probability G fools D)


""" Feature Loss """
#VGG
#content = imread('abbeyexample_copy.png')/256
#content = gray2rgb(rgb2gray(content))
# shape = (1,256,256,3)
shape = (mb_size, 256, 256, 3)
pooling = 'avg'
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
network = 'imagenet-vgg-verydeep-19.mat'
vgg_weights, vgg_mean_pixel = vgg.load_net(network)         
print(5)
orig_image = tf.placeholder('float', shape = shape)  #need to feed it with (1,256,256,3) objects
print(orig_image)
orig_content = vgg.preprocess(orig_image, vgg_mean_pixel)  #tensor (1,256,256,3)
print(orig_content)
print('G_sample.shape',G_sample.shape)
#G_sample = tf.reshape(G_sample,(mb_size,256,256))
#G_sample = tf.stack([G_sample,G_sample,G_sample],axis=3)   #tensor (256,256,3)
#print('G_sample.shape',G_sample.shape)
gen_content = vgg.preprocess(G_sample, vgg_mean_pixel)
# gen_content = tf.expand_dims(gen_content,0)
print('ok')
orig_net = vgg.net_preloaded(vgg_weights, orig_content, pooling)
print('ok1')
gen_net = vgg.net_preloaded(vgg_weights, gen_content, pooling)
#content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
#print('content_pre.shape',content_pre.shape)
print(6)

""" Losses """
#feat_loss
feat_loss = 0
for layer in CONTENT_LAYERS:
    feat_loss += tf.nn.l2_loss(orig_net[layer] - gen_net[layer])

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# labels = matrice de 1, même type que logits
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))) + coeff*feat_loss
# G_loss est basée sur D_logit_fake !

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0 
zerotime = time.time()
st = time.time()
index_in_epoch = 0
D_loss_list = []
G_loss_list = []


saver = tf.train.Saver({
    "D_W1": D_W1, "D_W2": D_W2, "D_b1": D_b1, "D_b2": D_b2,
    "G_W1": G_W1, "G_W2": G_W2, "G_b1": G_b1, "G_b2": G_b2})
print('orig_image', orig_image)


for it in range(0,num_it):


    if it>0:
        last_X = X_mb
    X_mb = []
    Y_mb = [0 for l in range(0,mb_size)]
    #for k in range(0,mb_size):
    #    a = np.random.randint(0,L_data-1)
    #    X_mb.append(data[a])
    for k in range(0,mb_size):              # 'sliding window'
        X_mb.append(data[(it+k)%L_data])
    X_mb = np.stack(X_mb,axis=0)
    #X_mb = np.vstack(X_mb)
    #print('X ',X_mb.shape)
    
    
    for p in range(0,len(X_mb)):         
    #    print('0',X_mb[p].shape)
        Y_mb[p] = X_mb[p,:,:]
    #    print('1',Y_mb[p].shape)
    #    print('Y[p] ',Y_mb[p].shape)
        Y_mb[p] = np.stack([Y_mb[p],Y_mb[p],Y_mb[p]],axis=2)
    #    print('2',Y_mb[p].shape)
    
    Y_mb = np.stack(Y_mb,axis=0)
    #print('Y ',Y_mb.shape)
    
    if it % 100 == 0 and it % 1000 != 0:
        print(it)

    if  it % 1000 == 0:

        delta = time.time() - st
        st = time.time()
        # n_sample = 1
        #if it>0:
        #    last_X = X_mb 
        #X_mb = []
        #Y_mb = []
        #for k in range(0,mb_size):
        #    a = np.random.randint(0,L_data-1)
        #    X_mb.append(data[a])
        #X_mb = np.vstack(X_mb)

        if it>0:
            samples = sess.run(G_sample, feed_dict={Z: X_mb})
            #print('\n samples.shape',samples.shape)
            fig = plot(samples,last_X)
            plt.savefig('./newdataset/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)
        i += 1


    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: X_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: X_mb, orig_image: X_mb})


    if it % 1000 == 0:

        print('\n   Iter: {}'.format(it))
        print('D_loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr),'( includes feat_loss )')
        D_loss_list.append(D_loss_curr)
        G_loss_list.append(G_loss_curr)
        plotloss2y(D_loss_list,G_loss_list,'./newdataset/plotloss.jpg')
        #print('y = ',y)
        if it != 0 and it != num_it:     # temps d'execution
            t = (st - zerotime) * (num_it - it) / it
            print('time since last printing : {:.4} '.format(delta),'sec')
            print('ends approximately in : {:.4} '.format(t),'sec')
            print('(',int(t/60),'min',int((t/60-int(t/60))*60),'sec )')
    #print('y_sample = ',y_sample)
    #print('samples[1,:].shape = ',samples[1,:].shape)
    #print('samples[1,:] = ',samples[1,:])
    #print('samples.shape = ',samples.shape)
    #print('samples[1,:] sum = ',sum(samples[1,:]))
    #print('samples[:,1] sum = ',sum(samples[:,1]))
    #print('X_mb.shape = ',X_mb.shape)
    #print('y_mb.shape = ',y_mb.shape)
    #print('y_mb = ',y_mb)   # y_mb = labels de toutes les images (64) du batch => [0,0,0,0,0,0,0,1,0,0] = label 7
        print()

save_path = saver.save(sess, "out/model.ckpt")
print("Model saved in file: %s" % save_path)

sess.close()
