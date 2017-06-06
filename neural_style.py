# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc

from stylize import stylize

import math
from argparse import ArgumentParser

from PIL import Image

# default arguments
CONTENT_WEIGHT = 5e0  #5e0 (=> importance du contenu par rapp. au style)      (= ALPHA/2)
CONTENT_WEIGHT_BLEND = 1     #prise en compte des différentes images
STYLE_WEIGHT = 5e2    #5e2 (=> importance du style par rapp. au contenu)      (= k*BETA)
TV_WEIGHT = 1e2
MATTE_WEIGHT = 1e1
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.89    #0,89 (BETA1 diminue => trop netteté, flou diminue, couleurs !=)
BETA2 = 0.999     #0,999
EPSILON = 1e-08
STYLE_SCALE = 1.0    #taille de style
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'    #average pooling (avg) meilleur que max car améliore le gradient flow
#WIDTH=None    (WIDTH=content_image.shape[1])


print("\n")

def build_parser():       #definition optional arguments
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--matte',
            dest='matte', help='matte laplacian',
            metavar='MATTE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path (.jpg)',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')                                        #fréquence à laquelle on affiche les loss instantannées
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--matte-weight', type=float,
            dest='matte_weight', help='laplacian matte weight (default %(default)s)',
            metavar='MATTE_WEIGHT', default=MATTE_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
            dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
            metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
            dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    parser.add_argument('--dest-fig',
            dest='dest_fig', help="name of the graph's file ",
            metavar='FIGURE DESTINATION')
    parser.add_argument('--dest-txt',
            dest='dest_txt', help="name of the text's file ",
            metavar='TEXT DESTINATION')
    
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):             #Test Existence du réseau
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imread(options.content)
    style_images = [imread(style) for style in options.styles] #Si plusieurs images de style
    print("content_image.shape ",content_image.shape)
    i=0
    for style in options.styles:
        print("style_image n.",i+1," ",style_images[i].shape)
        i=i+1
        
    width = options.width
    print("width",width)
    if width is not None:                                               #Modifie la taille de l'image de base
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        print("newshape",new_shape)
        content_image = scipy.misc.imresize(content_image, new_shape) #Remplace les 2 1ers éléments de content_image.shape par ceux de new_shape
    target_shape = content_image.shape
    print("targetshape",target_shape)
    
    for i in range(len(style_images)): #pour chaque image de style, change format de l'array
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        print("style_scale",style_scale)
        print("style_images[i].shape[1]",style_images[i].shape[1])
        print("coeff", style_scale *
                target_shape[1] / style_images[i].shape[1])
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])     #Multiplie les 2 1ers éléments de style_images[i].shape par coeff
        print("style_images[",i,"].shape",style_images[i].shape)
        

    style_blend_weights = options.style_blend_weights #liste
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images] #liste à n élts (n=nb d'images de style) = part de chaque image de style dans l'image finale
        print("(None) style_blend_weights",style_blend_weights)
    else:
        print("style_blend_weights",style_blend_weights)
        total_blend_weight = sum(style_blend_weights)
        print("total_blend_weight",total_blend_weight)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights] #=>sum(style_blend_weights)=1
        print("style_blend_weights",style_blend_weights)

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])
        print("initial.shape",initial.shape)
        # Initial guess is specified, but not noiseblend - no noise should be blended
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 0.0
    else:
        # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 1.0
        if options.initial_noiseblend < 1.0:
            initial = content_image
    print("options.initial_noiseblend",options.initial_noiseblend)
    print("\n")
    
    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    for iteration, image in stylize(
        network=options.network,
        initial=initial,
        initial_noiseblend=options.initial_noiseblend,
        content=content_image,
        styles=style_images,
        matte=options.matte,
        preserve_colors=options.preserve_colors,
        iterations=options.iterations,
        content_weight=options.content_weight,
        content_weight_blend=options.content_weight_blend,
        style_weight=options.style_weight,
        style_layer_weight_exp=options.style_layer_weight_exp,
        style_blend_weights=style_blend_weights,
        tv_weight=options.tv_weight,
        matte_weight=options.matte_weight,
        learning_rate=options.learning_rate,
        beta1=options.beta1,
        beta2=options.beta2,
        epsilon=options.epsilon,
        pooling=options.pooling,       
        output=options.output,
        dest_txt=options.dest_txt,
        dest_fig=options.dest_fig,        #ajout personnel
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations,
    ):
        output_file = None
        combined_rgb = image
        if iteration is not None:
            if options.checkpoint_output:                               #enregistre des images intermédiaires
                output_file = options.checkpoint_output % iteration     #nom des images intermédiaires
                
        else:                                                           
            output_file = options.output
        if output_file:
            imsave(output_file, combined_rgb)
        


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)   #lit le fichier image comme un array
    #imread(path, flatten=True) : Noir&Blanc (supprime couleurs des fichiers de départ)
    if len(img.shape) == 2:
        # grayscale (si image de type B&W)
        img = np.dstack((img,img,img)) #On met sous la forme RVB
    elif img.shape[2] == 4:      #3eme dimension de l'array : RVB+alpha (transparence)               
        # PNG with alpha channel
        img = img[:,:,:3]     #on supprime le parametre alpha
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    main()
