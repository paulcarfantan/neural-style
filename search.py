from argparse import ArgumentParser
from functools import reduce
import numpy as np
import os
import scipy.misc
import shutil
import tensorflow as tf
import vgg

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
IMG_X = 64
IMG_Y = 64

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content-dir', dest='content_dir', required=True,
                        help='content image directory')
    parser.add_argument('--style-dir', dest='style_dir', required=True,
                        help='style image directory')
    parser.add_argument('--n-style', dest='n_style', default=10, type=int,
                        help='number of style images to find')
    parser.add_argument('--output-dir', dest='output_dir', default="output",
                        help='output image directory')
    parser.add_argument('--recurse', dest='recurse', default=True,
                        help='Recurse through sub-directories while searching')
    parser.add_argument('--network', dest='network', default=VGG_PATH,
                        help='path to network parameters')
    return parser


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def read_img(fname):
    image = scipy.misc.imread(fname)
    if image.ndim == 2:
        image = gray2rgb(image)
    if image.ndim == 4:
        image = image[:,:,:,0]
    if image.shape[2] > 3:
        image = image[:,:,:3]
    image = scipy.misc.imresize(image, (IMG_X, IMG_Y, 3))
    return image


def score_img(content_features, fname, vgg_weights, vgg_mean_pixel):
    '''Return the scores for a given image, compared with content image

    '''
    try:
        style = read_img(fname)
    except:
        # disregard file if it cannot be read as an image
        return np.float('inf')

    scores = np.zeros(len(content_features))

    with tf.Graph().as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=(1,)+style.shape)
        net = vgg.net_preloaded(vgg_weights, image, 'max')
        style_pre = np.array([vgg.preprocess(style, vgg_mean_pixel)])
        for i, s in enumerate(scores):
            content_losses = []
            for content_layer in CONTENT_LAYERS:
                content_losses.append(tf.nn.l2_loss(
                    net[content_layer] - content_features[i][content_layer]))
            content_loss = reduce(tf.add, content_losses)
            scores[i] = content_loss.eval(feed_dict={image: style_pre})
    print(fname)
    return scores


def search_dir(content, vgg_weights, vgg_mean_pixel, style_score, style_file,
               base_dir, recurse):
    '''Search a directory for images

    Go through a directory and update the best style score and style file
    matrices based on the scores of all images in the directory. If recurse is
    true, go through any sub-directories to look for new images.
    '''

    print('Searching %s for images' % base_dir)
    for f in os.listdir(base_dir):
        fname = os.path.join(base_dir, f)
        if os.path.isdir(fname) and recurse:
            style_score, style_file = search_dir(
                content, vgg_weights, vgg_mean_pixel, style_score, style_file,
                fname, recurse)
        else:
            scores = score_img(content, fname, vgg_weights, vgg_mean_pixel)
            for i, score in enumerate(scores):
                maxi = style_score[i, :].argmax()
                maxs = style_score[i, maxi]
                if score < maxs:
                    style_score[i, maxi] = score
                    style_file[i, maxi] = fname
    return style_score, style_file


def main():
    '''Search for similar images

    Search the style directory for images that closely resemble each image in
    the content directory. Save those images in an output directory folder
    corresponding to each content image, renamed as their matching rank number.
    '''

    parser = build_parser()
    options = parser.parse_args()

    content_files = os.listdir(options.content_dir)
    content_images = [read_img(os.path.join(options.content_dir, f))
                      for f in content_files]

    # n_content by n_style matrix and list to store the best style images
    n_content = len(content_files)
    n_total = n_content * options.n_style
    best_style_score = np.float('inf') * np.ones((n_content, options.n_style))
    best_style_file = np.array([['' for i in range(options.n_style)]
                                for h in range(n_content)], dtype=object)

    vgg_weights, vgg_mean_pixel = vgg.load_net(options.network)

    content_features = [{} for _ in content_images]
    for i, c in enumerate(content_images):
        with tf.Graph().as_default(), tf.Session() as sess:
            image = tf.placeholder('float', shape=(1,)+c.shape)
            net = vgg.net_preloaded(vgg_weights, image, 'max')
            content_pre = np.array([vgg.preprocess(c, vgg_mean_pixel)])
            for layer in CONTENT_LAYERS:
                content_features[i][layer] = net[layer].eval(
                    feed_dict={image: content_pre})

    final_style_score, final_style_file = search_dir(
        content_features, vgg_weights, vgg_mean_pixel, best_style_score,
        best_style_file, options.style_dir, options.recurse)

    if np.any(np.isinf(final_style_score)):
        inf_total = np.sum(np.isinf(final_style_score))
        print('%d out of %d style images not found.' % (inf_total, n_total),
              'Try rerunning with a smaller n-style.')
        raise

    sorted_files = final_style_file[
        np.indices((n_content, options.n_style))[0],
        final_style_score.argsort()]

    format_str = '{0:0>%d}.{1}' % np.ceil(np.log10(n_total))

    os.mkdir(options.output_dir)
    for i, f in enumerate(content_files):
        fname = ''.join(f.split('.')[:-1])
        print('Copying style files for %s' % fname)
        os.mkdir(os.path.join(options.output_dir, fname))
        for j in range(options.n_style):
            img_ext = sorted_files[i, j].split('.')[-1]
            shutil.copy(sorted_files[i, j], os.path.join(
                options.output_dir, fname, format_str.format(j, img_ext)))


if __name__ == '__main__':
    main()
