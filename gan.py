import tensorflow as tf
import numpy as np
import os
import vgg
import time
from PIL import Image
import scipy.misc
from scipy.misc import imread, imresize

# TODO: put util functions in common utils.py file
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

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

Z_dim = 256
X_dim = 256
h_dim = 256

# Discriminator - TODO: must be same as in source
X = tf.placeholder(tf.float32, shape=[X_dim, X_dim])
D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[X_dim, h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[h_dim, 1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(inputs):
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

# Generator - TODO: must be same as in source
G_W1 = tf.Variable(xavier_init([Z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[Z_dim, h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[Z_dim, X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

saver = tf.train.Saver({
    "D_W1": D_W1, "D_W2": D_W2, "D_b1": D_b1, "D_b2": D_b2,
    "G_W1": G_W1, "G_W2": G_W2, "G_b1": G_b1, "G_b2": G_b2})

def improcess(image):
    return np.reshape(rgb2gray(imresize(imread(image),(256,256))),(X_dim, X_dim))

input_image = improcess("examples/1-content.jpg")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print("Random weights: ")
  D_prob, D_logit = sess.run(discriminator(X), feed_dict={X: input_image})
  print(D_prob, D_logit)
  saver.restore(sess, "out/model.ckpt")

  print("Saved weights: ")
  D_prob, D_logit = sess.run(discriminator(X), feed_dict={X: input_image})
  print(D_prob, D_logit)
