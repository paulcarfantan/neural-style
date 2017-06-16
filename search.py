from argparse import ArgumentParser
import math
import numpy as np
import os
from PIL import Image
import scipy.misc
import shutil

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content-dir', dest='content_dir', required=True,
                        help='content image directory')
    parser.add_argument('--style-dir', dest='style_dir', required=True,
                        help='style image directory')
    parser.add_argument('--n-style', dest='n_style', default=10,
                        help='number of style images to find')
    parser.add_argument('--output-dir', dest='output_dir', default="output",
                        help='output image directory')
    parser.add_argument('--recurse', dest='recurse', default=True,
                        help='Recurse through sub-directories while searching')
    return parser


def score_img(content, fname):
    '''
    Return the scores for a given image, compared with content image
    '''
    return np.random.rand(len(content))


def search_dir(content, style_score, style_file, base_dir, recurse):
    '''
    Go through a directory and update the best style score and style file
    matrices based on the scores of all images in the directory. If recurse is
    true, go through any sub-directories to look for new images.
    '''

    print('Searching %s for images' % base_dir)
    for f in os.listdir(base_dir):
        fname = os.path.join(base_dir, f)
        if os.path.isdir(fname) and recurse:
            style_score, style_file = search_dir(
                content, style_score, style_file, fname, recurse)
        else:
            scores = score_img(content, fname)
            for i, score in enumerate(scores):
                maxi = style_score[i, :].argmax()
                maxs = style_score[i, maxi]
                if score < maxs:
                    style_score[i, maxi] = score
                    style_file[i, maxi] = fname
    return style_score, style_file


def main():
    '''
    Search the style directory for images that closely resemble each image in
    the content directory. Save those images in an output directory folder
    corresponding to each content image, renamed as their matching rank number.
    '''

    parser = build_parser()
    options = parser.parse_args()

    content_files = os.listdir(options.content_dir)
    content_images = [scipy.misc.imread(os.path.join(options.content_dir,f))
                      for f in content_files]

    # n_content by n_style matrix and list to store the best style images
    n_content = len(content_files)
    n_total = n_content * options.n_style
    best_style_score = np.float('inf') * np.ones((n_content, options.n_style))
    best_style_file = np.array([['' for i in range(options.n_style)]
                                for h in range(n_content)], dtype=object)

    final_style_score, final_style_file = search_dir(
        content_images, best_style_score, best_style_file, options.style_dir,
        options.recurse)

    if np.any(np.isinf(final_style_score)):
        inf_total = np.sum(np.isinf(final_style_score))
        raise Error(('%d out of %d style images not found.'
                     % (inf_total, n_total)) +
                    'Try rerunning with a smaller n-style.')

    asort = final_style_score.argsort()
    sorted_files = final_style_file[np.indices((n_content, options.n_style))[0],
                                    final_style_score.argsort()]

    format_str = '{0:0>%d}.{1}' % np.ceil(np.log10(n_total))

    os.mkdir(options.output_dir)
    for i, f in enumerate(content_files):
        fname = ''.join(f.split('.')[:-1])
        print('Copying style files for %s' % fname)
        os.mkdir(os.path.join(options.output_dir, fname))
        for j in range(options.n_style):
            img_ext = sorted_files[i,j].split('.')[-1]
            shutil.copy(sorted_files[i,j], os.path.join(
                options.output_dir, fname, format_str.format(j, img_ext)))


if __name__ == '__main__':
    main()
