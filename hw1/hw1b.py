from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
from os import listdir
import os
rng = np.random

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''
    f, axarr = plt.subplots(4, 4)
    print D.shape
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            axarr[i, j].imshow(D[:, idx].reshape(sz,sz), cmap = cm.Greys_r)
    f.savefig(imname)
    plt.close(f)


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    result = np.asarray(np.dot(c.T, D.T)) #1 * N
    result = result.reshape(256, 256)
    print result.shape
    result += X_mn
    ax.imshow(result, cmap = cm.Greys_r)


def read_imge_dir_to_arr(dirname):
    onlyfiles = [os.path.join(dirname, f) for f in listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    onlyfiles.sort()
    num_of_imgs = len(onlyfiles)
    width, height = Image.open(onlyfiles[0]).size
    return np.stack([np.asarray(list(Image.open(file_str).getdata())).reshape(width * height) for file_str in onlyfiles])

if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''
    Ims = read_imge_dir_to_arr("jaffe")
    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues
    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''

    iter_step = 20
    num_of_eigens = 16
    num_of_imgs = X.shape[0]
    update_rate = 0.1
    width = 256
    height = 256
    D = np.zeros((width * height, num_of_eigens), dtype=np.float)
    lamb = np.zeros(num_of_eigens, dtype=np.float)
    for i in range(num_of_eigens):
        d = theano.shared(rng.randn(width * height), name="d")
        X_d = T.dot(X, d)
        #get loss
        loss = T.dot(X_d.T, X_d)
        for j in range(0, i):
            d_j_d = T.dot(D[:,j].T, d)
            loss = loss - lamb[j] * T.dot(d_j_d.T, d_j_d)
        loss = -loss
        #gradient
        g_d = T.grad(loss, [d])[0]
        #training function
        train = theano.function(
            inputs=[],
            outputs = [d, loss],
            updates=[(d, (d - update_rate * g_d) / (d - update_rate * g_d).norm(L=2))]
            )
        res_d = None
        #iteration
        for j in range(iter_step):
            res_d, per_loss = train()
        D[:,i] = res_d
        lamb[i] = np.dot(np.dot(np.dot(res_d.T, X.T),X), res_d)
        print lamb[i]

    c = np.dot(D.T, X.T)
    print "D c calculate done"
    for i in range(0,200,10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')
