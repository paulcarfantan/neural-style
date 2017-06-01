from __future__ import division
from argparse import ArgumentParser
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.misc
import scipy.sparse
from scipy.sparse import csr_matrix
from skimage.color import rgb2gray


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--image', dest='image',
                        help='content image',
                        type=str, required=True)
    parser.add_argument('--scribbled', dest='scribbled',
                        help='scribbled image',
                        type=str, required=False)
    parser.add_argument('--lambda', dest='lambda_c',
                        help='lambda confidence value',
                        type=int, default=100, required=False)
    parser.add_argument('--epsilon', dest='epsilon',
                        help='epsilon smoothness value',
                        type=float, default=10**(-7), required=False)
    return parser


def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def computeLaplacian(img, epsilon, win_rad=1):
    '''Returns sparse matting laplacian'''
    win_size = (win_rad*2+1)**2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h-win_rad-1, w-win_rad-1
    win_diam = win_rad*2+1

    indsM = np.arange(h*w).reshape((h, w))
    ravelImg = img.reshape(h*w, d)
    win_inds = rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=2, keepdims=True)
    win_var = (np.einsum('...ji,...jk ->...ik', winI, winI)/win_size
               - np.einsum('...ji,...jk ->...ik', win_mu, win_mu))

    inv = np.linalg.inv(win_var + (epsilon/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1/win_size)*(1 + np.einsum('...ij,...kj->...ik',
                                                          X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)),
                                shape=(h*w, h*w))
    return vals, L


def closed_form_matte(img, scribbled_img, lambda_c, epsilon):
    h, w, c = img.shape
    scribbles_loc_img = rgb2gray(scribbled_img) - rgb2gray(img)
    bgInds = np.where(scribbles_loc_img.ravel() < 0)[0]
    fgInds = np.where(scribbles_loc_img.ravel() > 0)[0]
    D_s = np.zeros(h*w)
    D_s[fgInds] = 1
    D_s[bgInds] = 1
    b_s = np.zeros(h*w)
    b_s[fgInds] = 1

    vals, L = computeLaplacian(img/255, epsilon)
    sD_s = scipy.sparse.diags(D_s)

    x = scipy.sparse.linalg.spsolve(L + lambda_c*sD_s, lambda_c*b_s)
    alpha = np.minimum(np.maximum(x.reshape(h, w), 0), 1)
    return vals, L, alpha


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def compute_loss(img, matte):
    loss = 0.0
    for i in range(3):
        loss += img[:,:,i].ravel().transpose().dot(
            matte.dot(img[:,:,i].ravel()))
    return loss


def main():
    parser = build_parser()
    options = parser.parse_args()
    img = scipy.misc.imread(options.image)
    if img.shape[2] > 3:
        img = img[:,:,0:3]
    title, ext = options.image.split(".")
    name = title.split("/")[-1]
    if options.scribbled:
        scribbled_img = scipy.misc.imread(options.scribbled)
    else:
        scribbled_img = scipy.misc.imread(title+"_m."+ext)
    # If Memory Error, use resize:
    # img = scipy.misc.imresize(img, 0.25)
    # scribbled_img = scipy.misc.imresize(scribbled_img, 0.25)
    vals, L, alpha = closed_form_matte(
        img, scribbled_img, options.lambda_c, options.epsilon)
    save_sparse_csr("data/"+name, L.tocsr())
    scipy.misc.imsave(title+"_alpha."+ext, alpha)
    print("Computing base loss for ", name, " with size ", img.shape)
    loss = compute_loss(img, L)
    print("Loss: ", loss)

if __name__ == "__main__":
    main()
