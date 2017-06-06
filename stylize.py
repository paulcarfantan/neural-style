# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.
"""
+=  -=  *=  /=
c=a
c+=b
=> c=a+b


content_weight = alpha/2 ?
"""

import vgg

import tensorflow as tf
import numpy as np

from scipy.sparse import csr_matrix
from sys import stderr

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
dest_fig='figure.jpg'
try:
    reduce                           #différentes versions de python => peut-être besoin appeler functools
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, matte,
        preserve_colors, iterations, content_weight, content_weight_blend,
        style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        matte_weight, learning_rate, beta1, beta2, epsilon, pooling,
        output, dest_txt, dest_fig,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded       
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape                               #rajoute un 1 en tant que 1ere dimension de content
    style_shapes = [(1,) + style.shape for style in styles]    #idem sur les images de style 
    content_features = {}                                      #Création dico 
    style_features = [{} for _ in styles]                      #idem pour chaque image de style 

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)         

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight  # => relu1_1 : 1  ;  relu2_1 : 1*style_layer_weight_exp  ; ... ;  relu5_1 : (style_layer_weight_exp)**4
        layer_weight *= style_layer_weight_exp            # (default : style_layer_weight_exp=1) => seulement des 1

    # normalize style layer weights => sum=1
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum    # => on obtient 1 liste normalisée à 5 élts pour chaque image de style

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:   #Toutes les opérations construites dans ce contexte (indentées) seront placées sur le CPU:0 et dans le graphe g
                                                                     #"with Session" ferme la session lorsque c'est terminé       
        image = tf.placeholder('float', shape = shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)         #dictionnaire associant à chaque élt de VGG19-LAYERS un tensor , shape.len=4
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})
        

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])   #retourne une matrice image_style[i] - vgg_mean_pixel
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
#                print("\n")
#                print(features)
#                print("shape",features.shape, features.size)
#                print("\n")
                features = np.reshape(features, (-1, features.shape[3]))
#                print("\n")
#                print(features)
#                print("shape",features.shape, features.size)
#                print("\n")
                gram = np.matmul(features.T, features) / features.size   #matmul = matrix multiplication  => gram=[features(transposée) x features] / features.size
                style_features[i][layer] = gram                          #style_features = liste de dictionnaires

    initial_content_noise_coeff = 1.0 - initial_noiseblend     #noiseblend = input (optionnel)

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        #initial = tf.random_normal(shape) * 0                              #image de départ = blanche

        if initial is None:                                                     #initial = image de laquelle on part pour construire l'image suivante
            #noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256                           #initial non renseignée => aléatoire                    
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])       # initial - mean_pixel
            initial = initial.astype('float32')
            #noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)  
        #(default : initial_noiseblend=0) => initial = inchangé
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)


        # content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend      #default : content_weight_blend = 1      ==>...['relu4_2]=1      
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend                                        #==>...['relu5_2]=0

        content_loss = 0          #initialisation inutile mais on garde le même format pour style loss
        content_losses = []
        for content_layer in CONTENT_LAYERS:              #CONTENT_LAYERS = ('relu4_2', 'relu5_2')
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(             #content_weight = alpha/2
                    net[content_layer] - content_features[content_layer]) / content_features[content_layer].size))         #content_losses = liste de 2 élts
                    #net[content_layer] = features de l'image générée ; content_features[content_layer] = features de l'image d'origine 
        content_loss += reduce(tf.add, content_losses)       # = somme des élts de content_losses (on calcule l'erreur sur chaque layer, puis on additionne ces erreurs)
#(default : content_layers_weights['relu5_2]=0 => content_loss = content_losses[0])

        # style loss
        style_loss = 0
        for i in range(len(styles)):       #nb d'images de style
            style_losses = []
            for style_layer in STYLE_LAYERS:             #STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
                layer = net[style_layer]
                _, height, width, number = map(lambda j: j.value, layer.get_shape())   # "_" => discard the first elt of the tuple
                        #lambda = definit la fonction qui à j associe j.value ; map applique la fonction à tous les élts de layer.get_shape() ; (get_shape() = shape pour les tf)
                size = height * width * number
#                print("number ",number)
#                print("layer.shape",layer.get_shape())
                feats = tf.reshape(layer, (-1, number))      #supprime dim0 (=1), dim0=dim1*dim2, dim1=dim3=number      => shape = (dim1*dim2 , number)
#                print("feats.shape",feats.get_shape())
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[i][style_layer]                  #style_features = liste de dictionnaires initialisée dans "compute style features in feedforward mode"
                style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size) #liste contenant les erreurs de tous les layers de l'image i
                #gram = style representation of generated image ; style_gram = style representation of original image 
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)   
            #incrémentation de style_loss : reduce=sum(err layers de im[i]) ; style_weight = poids du style par rapp au content ; style_blend_weights[i] = poids de l'im. i par rapp aux autres
            # += => on somme les losses de toutes les images

        # matting lapacian loss
        loader = np.load(matte)
        lcoo = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                        shape=loader['shape']).tocoo()
        lindices = np.mat([lcoo.row, lcoo.col]).transpose()
        lvalues = tf.constant(lcoo.data,  dtype=tf.float32)
        laplacian = tf.SparseTensor(indices=lindices, values=lvalues, dense_shape=lcoo.shape)

        matte_loss = 0
        matte_losses = []
        for i in range(3):
            imr = tf.reshape(image[:,:,:,i], [-1, 1])
            matte_losses.append(
                tf.matmul(tf.transpose(imr),
                          tf.sparse_tensor_dense_matmul(laplacian, imr))[0][0]
            )
        matte_loss += matte_weight * reduce(tf.add, matte_losses)


        # total variation denoising                       (pas très importante : à remplacer par une autre loss ?)
        print("\n total variation denoising")            #(possible de désactiver la tv loss avec la commande --tv-weight 0)   

        tv_y_size = _tensor_size(image[:,1:,:,:])
        print(tv_y_size)
        tv_x_size = _tensor_size(image[:,:,1:,:])
        print(tv_x_size)
        print("\n")
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
                
        # overall loss
        loss = content_loss + style_loss + matte_loss + tv_loss         #total         #make alpha etc appear

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)       #opération qui met à jour les variables pour que total loss soit minimisé 
        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)         #ne marche pas                                      #quelles variables ???
        #train_step = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
        
        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('    matte loss: %g\n' % matte_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())
        
        
        
        # optimization
        best_loss = float('inf')                          #???
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())    #initialise les variables globales
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):     #Si on a rentré un pas pour print_iterations, on affiche avant la 1ere iteration les loss de initial
                print_progress()
                
            c_loss = []                #initialisation des listes de valeurs de loss
            s_loss = []
            t_loss = []
            tot_loss = []    
            x=[i+1 for i in range(iterations)]   #initialisation des abscisses des graphes
            
            for i in range(iterations):
                
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()                                               #on minimise loss à chaque itération

                c_loss.append(content_loss.eval())         #incrémentation des listes de valeurs de loss pour chaque itération
                s_loss.append(style_loss.eval())
                t_loss.append(tv_loss.eval())
                tot_loss.append(loss.eval())

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0)   : #i % print_iterations = reste de la diveucl de i par print_iterations      
                    print_progress()                                                 #On affiche les loss instantannées avec une fréquence = print_iterations
                    if last_step :
                        if dest_txt is None:
                            l=len(output)-4                #Création d'un fichier contenant les losses (même nom que l'output mais .txt)
                            file=output[:l]
                            F=open(''.join([file,'.txt']),'x')      #fusionne file et '.txt'
                            F.writelines(['  content loss: %g\n' % content_loss.eval() , '    style loss: %g\n' % style_loss.eval() , 
                                          '       tv loss: %g\n' % tv_loss.eval() , '    total loss: %g\n' % loss.eval()])
                            F.close
                        else:
                            F=open(dest_txt,'x')   
                            F.writelines(['  content loss: %g\n' % content_loss.eval() , '    style loss: %g\n' % style_loss.eval() , 
                                          '       tv loss: %g\n' % tv_loss.eval() , '    total loss: %g\n' % loss.eval()])
                            F.close
                        
                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:          
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()            #on associe l'image finale à la meilleure loss totale

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)
                    
                    if preserve_colors and preserve_colors == True:                           #preserve-colors
                        original_image = np.clip(content, 0, 255)        #clip = tous les élts de content >255 -->255, idem <0 -->0
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))


                    yield (
                        (None if last_step else i),
                        img_out
                    )
            
            
            #Nom de la destination des courbes
            if dest_fig is None :               
                l=len(output)-4                
                file=output[:l]
                dest_fig=''.join([file,'_fig','.jpg'])
                
            print('dest_fig',dest_fig)


            #Tracé des graphes
            plt.figure(1)
            plt.title("Différents types d'erreurs - graphe classique et graphe semi-logarithmique")
            plt.subplot(2,1,1)
            plt.plot(x, c_loss, label='content_loss')
            plt.plot(x, s_loss, label='style_loss')
            plt.plot(x, t_loss, label='tv_loss')
            plt.plot(x, tot_loss, label='total_loss')
            plt.grid('on')
            plt.axis('tight')
            plt.legend()
            plt.ylabel('erreur')
            
            plt.subplot(2,1,2)
            plt.semilogy(x, c_loss, label='content_loss')
            plt.semilogy(x, s_loss, label='style_loss')
            plt.semilogy(x, t_loss, label='tv_loss')
            plt.semilogy(x, tot_loss, label='total_loss')
            plt.grid('on')
            plt.axis('tight')                         
            plt.xlabel("i (Nombre d'itérations)")
            plt.ylabel('erreur')
            plt.savefig(dest_fig)
            
       
                

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)      #multiplication par 1 de tous les arguments de get_shape()     ???

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

"""
#BRUIT => JAMAIS 2 FOIS LA MÊME IMAGE / LOSS MÊME AVEC LES MÊMES PARAMÈTRES À CAUSE DE L'IMAGE INITIALE ET DU BRUIT ( = RANDOM )
#TEST : SUPPRESSION DU CARACTÈRE ALÉATOIRE ( INITIAL = BLANC )          =>? MÊME IMAGE - MÊMES ERREURS - MÊMES GRAPHIQUES ???    NON !!! => Optimizer induit erreurs ?
#TEST : PLUSIEURS IMAGES STYLE A LA FOIS ?                                 OK mais grosses losses
#TEST : EXECUTION EN ENLEVANT CERTAINS LOSSES                              OK (petites losses mais normal car on en supprime)
#TEST : CHANGER LAYERS (RELU4_2 ; RELU5_2) LINES 119-120                  ???
#TEST : CHANGER OPTIMISEUR TENSORFLOW (EX: GRADIENTDESCENTOPTIMIZER)       :     ne marche pas (best.reshape : 'NoneType' object has no attribute 'reshape')    (=normal !)
"""
