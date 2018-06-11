import numpy as np
import h5py
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import time
from dnn_app_utils_v2 import sigmoid, relu, sigmoid_backward, relu_backward

tic = time.time()

param = {}
layer_dims = [12288, 20, 7, 5, 1]

W4 = np.loadtxt('W4.txt')
b4 = np.loadtxt('b4.txt')

W4_reshaped = np.zeros((layer_dims[4],layer_dims[3]))
b4_reshaped = np.zeros((layer_dims[4],1))

for j in range(W4_reshaped.shape[1]):
    W4_reshaped[0][j] = W4[j]

for j in range(b4_reshaped.shape[1]):
    b4_reshaped[0][j] = b4

def b_reshape(b_orig):
    b_reshaped = np.zeros((b_orig.shape[0],1))

    for i in range(b_reshaped.shape[0]):
        b_reshaped[i][0] = b_orig[i]

    return b_reshaped

for i in range(1,4):
    param['W' + str(i)] = np.loadtxt('W' + str(i) + '.txt')
    param['b' + str(i)] = b_reshape(np.loadtxt('b' + str(i) + '.txt'))
    param['W4'] = W4_reshaped
    param['b4'] = b4_reshaped

def linear_forward(A, W, b):

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches

def predict(X, parameters):

    m = X.shape[1]
    p = np.zeros((1, m))

    AL, caches = L_model_forward(X, parameters)

    for i in range(m):
        if AL[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p

my_image = "c4.jpg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "C:/Users/Ranjith/Desktop/ROS and AI/Cats and Non-Cats/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((64*64*3,1))
my_image = my_image/255.

my_prediction = predict(my_image, param)

if my_prediction == 1:
    print("It's a cat!")
elif my_prediction == 0:
    print("It's not a cat!")

toc = time.time()
print('Time taken: ' + str((toc - tic)) + "s", "\n")
