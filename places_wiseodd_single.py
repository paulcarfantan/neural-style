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


L_data = 1000
mb_size = 10  # 64                   
Z_dim = 256**2                                    
#X_dim = mnist.train.images.shape[1]    # ex: 784=28x28
X_dim = 256**2
h_dim = 128

sess = tf.Session()
 
print(1)
places = os.listdir('./abbey/')
data=[]
for i in range(L_data):
    #data.append(np.reshape(rgb2gray(imread('./abbey/' + places[i])),(1,X_dim)))
    data.append(np.reshape(rgb2gray(imread('./abbey/' + places[i])),(1,X_dim))/256)
print(data[0])

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)    #standard deviation of the normal distribution
    return tf.random_normal(shape=size, stddev=xavier_stddev)  #donne des valeurs aléatoires d'une distribution normale

print(2)
""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
#y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
# X_dim + y_dim plutôt que X_dim comme ça on train non seulement à dire si l'image appartient ou non au dataset original, MAIS AUSSI dire à quel label elle correspond !
D_b1 = tf.Variable(tf.zeros(shape=[None,h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[None,1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]  # Parameters to optimize
# => initial : D_W = random & D_b = zeros ; then = optimize to get best discriminator
# D_W1 : shape=(794,128)  ;  D_W2 : shape=(128,1)  Weights pour passer d'un layer à l'autre  (input -->D_W1--> relu -->D_W2--> result)
# D_b1 : shape=(128,)  ;  D_b2 : shape=(1,)    Biases

print(3)
def discriminator(x):                                 # Single layer network 
    #inputs = tf.concat(axis=1, values=[x, y])
    inputs=x
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)    # = max( 0 , inputs × D_W1 + D_b1 )      
    D_logit = tf.matmul(D_h1, D_W2) + D_b2               # include weights and biases
    D_prob = tf.nn.sigmoid(D_logit)                      # to have a result between 0 and 1 (probability)

    return D_prob, D_logit    # /!\ not scalar : operations /!\


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, h_dim])) 
G_b1 = tf.Variable(tf.zeros(shape=[None,h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[None,X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


#G_W1 : shape=(110,128)  ;  G_W2 : shape=(128,784)
#G_b1 : shape=(128,)  ;  G_b2 : shape=(784,)
print(4)

def generator(z):
    inputs=z
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    print("input:", inputs.shape, "G_log:", G_log_prob.get_shape().as_list(), "G_prob:", G_prob.get_shape().as_list())
    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(sample):
    #for i, sample in enumerate(samples):
        #plt.imshow(sample.reshape(256,256), cmap='Greys_r')
    fig = plt.figure()

    #ax = plt.gca()
    #plt.axis('off')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #ax.set_aspect('equal')
    plt.imshow(sample,cmap='Greys_r')
    return fig

G_sample = generator(Z)    # Z = random input images to begin with.
D_real, D_logit_real = discriminator(X)   # X = real dataset
D_fake, D_logit_fake = discriminator(G_sample)  # G_sample = generated dataset (fake)
# D_real & D_fake = unused   (D_fake = probability G fools D)


""" Feature Loss """
#VGG
#content = imread('abbeyexample_copy.png')/256
#content = gray2rgb(rgb2gray(content))
shape = (1,256,256,3)
pooling = 'avg'
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
network = 'imagenet-vgg-verydeep-19.mat'
vgg_weights, vgg_mean_pixel = vgg.load_net(network)         
print(5)
orig_image = tf.placeholder('float', shape = shape)
orig_content = vgg.preprocess(orig_image, vgg_mean_pixel)
print('G_sample.shape',G_sample.shape)
G_sample = tf.reshape(G_sample,(256,256))
G_sample = tf.stack([G_sample,G_sample,G_sample],axis=2)
print('G_sample.shape',G_sample.shape)
gen_content = np.array([vgg.preprocess(G_sample, vgg_mean_pixel)])
print('ok')
orig_net = vgg.net_preloaded(vgg_weights, sess.run(orig_content.eval(session=sess), pooling))
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
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))) # + feat_loss
# G_loss est basée sur D_logit_fake !

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0 
zerotime = time.time()
st = time.time()
num_it = 10000
index_in_epoch = 0

saver = tf.train.Saver({
    "D_W1": D_W1, "D_W2": D_W2, "D_b1": D_b1, "D_b2": D_b2,
    "G_W1": G_W1, "G_W2": G_W2, "G_b1": G_b1, "G_b2": G_b2})

for it in range(num_it):
    if it % 1000 == 0:

        delta = time.time() - st
        st = time.time()
        n_sample = 1

        #Z_sample = sample_Z(n_sample, Z_dim)   # random matrix shape (16,100)
       # y_sample = np.ones(shape=[n_sample, y_dim]) * 0.1
       # y_sample = np.zeros(shape=[n_sample, y_dim])
       # y_sample[:, 8] = 1
       # y_sample=np.random.rand(n_sample, y_dim)
       # y_sample=np.transpose(y_sample)/np.sum(y_sample,axis=1)
       # y_sample=np.transpose(y_sample)
       # y_sample = np.zeros(shape=[n_sample, y_dim])
       # for k in range(n_sample): 
       #     y_sample[k,np.random.randint(0,y_dim)]=1

        sample = sess.run(G_sample, feed_dict={Z: data[i]})    # samples.shape = (256,256)
        print('\n sample.shape',sample.shape)
        fig = plot(sample)
        plt.savefig('./newdataset/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
    
    
    def next_batch(batch_size,index_in_epoch):
        start = index_in_epoch
        if start + batch_size > L_data:
            np.random.shuffle(data)
            start = 0
            index_in_epoch = batch_size
        else:
            index_in_epoch += batch_size
        end = index_in_epoch
        return data[start:end], index_in_epoch
    
    X_mb, index_in_epoch = next_batch(mb_size,index_in_epoch)
    
    
    
#    b = it % (int(L_data/mb_size)-1)     # it congru a b modulo ...
#    if it >= L_data/mb_size : # (si pas assez d'éléments dans data pour finir le batch)
#        np.random.shuffle(data)

#    liste = [data[j] for j in range(b*mb_size,(b+1)*mb_size)]  # /!\ cas ou num_it*mb_size > nombre de samples (len(places) ?) /!\
#    X_mb = np.vstack(liste)     # shape : (mb_size, X_dim^2)

   # X_mb, _ = mnist.train.next_batch(mb_size)
   # y_mb = labels des images du dataset original

    #Z_sample = sample_Z(mb_size, Z_dim)
    
    print('content_pre.shape', content_pre.shape)
    #content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb[0], Z: X_mb[0]})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: X_mb[0]})
    # orig_image: X_mb[0]

    
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'.format(D_loss_curr),'( includes feat_loss )')
        print('G_loss: {:.4}'.format(G_loss_curr))
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
