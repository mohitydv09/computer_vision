import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main

def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    ## One hot encode the labels.
    label_train_encoded = np.eye(10)[label_train[0]].T

    ## Get shuffled indices.
    np.random.seed(37)  ## 33 gave results for MLP.
    random_indices = np.random.permutation(label_train.shape[1])

    ## Get the Mini Batches, numpy automatically takes care of the last batch
    mini_batch_x = [im_train[:,random_indices[i*batch_size:(i+1)*batch_size]] for i in range(math.ceil(im_train.shape[1]/batch_size))]
    mini_batch_y = [label_train_encoded[:,random_indices[i*batch_size:(i+1)*batch_size]] for i in range(math.ceil(im_train.shape[1]/batch_size))]
    
    return mini_batch_x, mini_batch_y

def fc(x, w, b):
    # TO DO
    y = w@x + b
    return y

def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dw = dl_dy @ x.T
    dl_db = dl_dy
    dl_dx = w.T @ dl_dy
    
    return dl_dx, dl_dw, dl_db

def loss_euclidean(y_tilde, y):
    # TO DO
    l = np.sum(np.square(y_tilde - y))
    dl_dy = 2*(y_tilde - y) 
    
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    ## Added for numerical stability.
    x_max = np.max(x)
    ## Apply Softmax layer.
    y_telda = np.exp(x-x_max)/(np.sum(np.exp(x-x_max)) + 1e-10)

    # Get the softmax loss.
    l = -np.sum(y*np.log(y_telda + 1e-10))

    dl_dy = y_telda - y

    return l, dl_dy

def relu(x):
    # TO DO
    x[x < 0] = 0
    y = x
    return y

def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dx = np.where(x > 0, dl_dy, 0)
    return dl_dx

def im2col(image, kernal_size, stride):
    # Assumes single channel input image.
    num_cols = ((image.shape[0]-kernal_size)//stride +1)*((image.shape[1]-kernal_size)//stride + 1)

    col_image = np.empty((kernal_size*kernal_size, num_cols))
    col_index = 0
    for col in range(0, image.shape[1]-kernal_size+1, stride):
        for row in range(0, image.shape[0]-kernal_size+1, stride):
            patch = image[row:row+kernal_size, col:col+kernal_size]
            col_image[:,col_index] = patch.flatten(order='F')
            col_index+=1
    return col_image

def conv(x, w_conv, b_conv):
    # TO DO

    ## Pad the imput image.
    x_padded = np.pad(x,((1,1),(1,1),(0,0)),mode='constant')  ## 16,16,1

    ## Create the col_image.
    kernal_size, stride = 3, 1
    col_image = im2col(x_padded.reshape((x_padded.shape[0],x_padded.shape[1])), kernal_size, stride)  ## 9,196 (Channel assumed one here, as the output is 2D)

    # Create flattened w.
    w = np.empty((3,9))
    for channel in range(w_conv.shape[3]):
        w[channel,:] = w_conv[:,:,0,channel].flatten(order='F')
 
    ## Perform the convolution.
    y = (w @ col_image + b_conv).T            ## 196, 3

    ## Reshape to image format.
    y = np.reshape(y,(14,14,3), order='F')
    
    return y

def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    ## Pad the input image.
    x_padded = np.pad(x, ((1,1),(1,1),(0,0)), mode='constant')

    ## Get the col Image.
    x_col = im2col(x_padded.reshape((x_padded.shape[0],x_padded.shape[1])),3,1)        ## 9,196

    ## Flatten the dl_dy.
    dl_dy_flat = dl_dy.reshape((14*14,3),order='F')    ## 196,3
    
    dl_dw = (dl_dy_flat.T)@(x_col.T)                  ## 3,9
    dl_dw = (dl_dw.T).reshape(w_conv.shape, order='F')

    ## Divide by 100 to get numerical stability.
    dl_db = np.sum(dl_dy, axis=(0,1)).reshape(b_conv.shape, order='F')/196

    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    stride = 2
    y = np.zeros((x.shape[0]//2, x.shape[1]//2, x.shape[2]))
    for channel in range(x.shape[2]):
        for col in range(0,x.shape[1]-1,stride):
            for row in range(0,x.shape[0]-1,stride):
                y[row//2, col//2, channel] = np.max(x[row:row+2, col:col+2, channel])
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    stride = 2
    dl_dx = np.zeros_like(x)
    for channel in range(x.shape[2]):
        for col in range(0,x.shape[1],stride):
            for row in range(0,x.shape[0],stride):
                patch = x[row:row+2, col:col+2,channel]
                idx = np.argmax(patch)
                max_row, max_col = np.unravel_index(idx, patch.shape)
                dl_dx[row+max_row,col+max_col,channel] = dl_dy[row//2,col//2,channel]
    return dl_dx

def flattening(x):
    # TO DO
    y = x.flatten(order='F').reshape((-1,1))
    return y

def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy.reshape(x.shape, order='F')
    return dl_dx

def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 1
    decay_rate = 0.5

    ## Initialize Weights.
    input_size = mini_batch_x[0].shape[0]
    output_size = mini_batch_y[0].shape[0]
    w = np.random.randn(output_size,input_size)
    b = np.random.randn(output_size,1)

    k = 0
    for iter in range(5001):
        if iter%1000 == 0:
            learning_rate = learning_rate*decay_rate
        dl_dw_batch = np.zeros_like(w)
        dl_db_batch = np.zeros_like(b)

        for x, y in zip(mini_batch_x[k].T, mini_batch_y[k].T):
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)

            ## Forward pass.
            y_tilde = fc(x, w, b)

            ## Compute Loss.
            loss, dl_dy = loss_euclidean(y_tilde, y)   ## dl_dy is columns

            ## Compute Gradients.
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)

            ## Add to the gradients of batch.
            dl_dw_batch += dl_dw
            dl_db_batch += dl_db

        k += 1
        if k == len(mini_batch_x):
            k = 0
        ## Update weights
        w -= (learning_rate/len(mini_batch_x[0]))*dl_dw_batch
        b -= (learning_rate/len(mini_batch_x[0]))*dl_db_batch
        
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 5
    decay_rate = 0.7

    ## Initialize Weights.
    input_size = mini_batch_x[0].shape[0]
    output_size = mini_batch_y[0].shape[0]
    w = np.random.randn(output_size,input_size)
    b = np.random.randn(output_size,1)

    k = 0
    for iter in range(5001):
        if iter%1000 == 0:
            learning_rate = learning_rate*decay_rate
        dl_dw_batch = np.zeros_like(w)
        dl_db_batch = np.zeros_like(b)
        for x, y in zip(mini_batch_x[k].T, mini_batch_y[k].T):
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)

            ## Forward pass.
            y_tilde = fc(x, w, b)

            ## Compute Loss.
            loss, dl_dy = loss_cross_entropy_softmax(y_tilde, y)   ## dl_dy is columns

            ## Compute Gradients.
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)

            ## Add to the gradients of batch.
            dl_dw_batch += dl_dw
            dl_db_batch += dl_db
            
        k += 1
        if k == len(mini_batch_x):
            k = 0
        ## Update weights
        w -= (learning_rate/len(mini_batch_x[0]))*dl_dw_batch
        b -= (learning_rate/len(mini_batch_x[0]))*dl_db_batch
        
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 5
    decay_rate = 0.99

    ## Initialize Weights.
    input_size = mini_batch_x[0].shape[0]
    output_size = mini_batch_y[0].shape[0]
    hidden_size = 30
    w1 = np.random.randn(hidden_size,input_size)
    b1 = np.random.randn(hidden_size,1)
    w2 = np.random.randn(output_size,hidden_size)
    b2 = np.random.randn(output_size,1)

    k = 0
    for iter in range(5001):
        if iter%1000 == 0:
            learning_rate = learning_rate*decay_rate
        dl_dw1_batch = np.zeros_like(w1)
        dl_db1_batch = np.zeros_like(b1)
        dl_dw2_batch = np.zeros_like(w2)
        dl_db2_batch = np.zeros_like(b2)

        for x, y in zip(mini_batch_x[k].T, mini_batch_y[k].T):
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)

            ## Forward pass hidden Layer.
            y_out_hidden = fc(x, w1, b1)   ## Output will be a 30 size column matrix
            ## Apply ReLU on this.
            y_out_relu = relu(y_out_hidden)
            ## Forward pass output layer
            y_out_last = fc(y_out_relu, w2, b2)

            ## Compute Loss.
            loss, dl_dy_last = loss_cross_entropy_softmax(y_out_last, y)   ## dl_dy is columns

            ## Compute Gradients last layer.
            dl_dx_hidden, dl_dw2, dl_db2 = fc_backward(dl_dy_last, y_out_hidden, w2, b2, y_out_last)
            ## Computer Gradients ReLU.
            dl_dx_relu = relu_backward(dl_dx_hidden, y_out_hidden, y_out_relu)
            ## Compute Gradients First Layer.
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dx_relu, x, w1, b1, y_out_hidden)

            ## Add to the gradients of batch.
            dl_dw2_batch += dl_dw2
            dl_db2_batch += dl_db2
            dl_dw1_batch += dl_dw1
            dl_db1_batch += dl_db1


        k += 1
        if k == len(mini_batch_x):
            k = 0
        ## Update weights
        w2 -= (learning_rate/len(mini_batch_x[0]))*dl_dw2_batch
        b2 -= (learning_rate/len(mini_batch_x[0]))*dl_db2_batch
        w1 -= (learning_rate/len(mini_batch_x[0]))*dl_dw1_batch
        b1 -= (learning_rate/len(mini_batch_x[0]))*dl_db1_batch   
    return w1, b1, w2, b2

def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 3
    decay_rate = 0.98

    ## Initialize Weights.
    kernal_size = 3
    channels_input = 1
    channels_output = 3
    flattened_size = 147
    output_size = 10
    w_conv = np.random.randn(kernal_size, kernal_size, channels_input, channels_output)
    b_conv = np.random.randn(channels_output,1)
    w_fc = np.random.randn(output_size,flattened_size)
    b_fc = np.random.randn(output_size,1)

    k = 0
    for iter in range(5001):
        # print("iter no : ", iter)
        if iter%1000 == 0:
            learning_rate = learning_rate*decay_rate

        dl_dw_conv_batch = np.zeros_like(w_conv)
        dl_db_conv_batch = np.zeros_like(b_conv)
        dl_dw_fc_batch = np.zeros_like(w_fc)
        dl_db_fc_batch = np.zeros_like(b_fc)


        for x, y in zip(mini_batch_x[k].T, mini_batch_y[k].T):
            x = x.reshape((14,14,1), order="F") ## Revert back to being a image.
            y = y.reshape(-1,1)

            ## Forward pass.
            y_conv = conv(x, w_conv, b_conv)   ## Output should be  : (14, 14, 3)
            y_relu = relu(y_conv)                                   # (14, 14, 3)
            y_maxpool = pool2x2(y_relu)                             # (7, 7, 3)                          
            y_flatten = flattening(y_maxpool)                       # (147, 1)
            y_fc = fc(y_flatten, w_fc, b_fc)                        # (10, 1)

            ## Compute Loss.
            loss, dl_dy_softmax = loss_cross_entropy_softmax(y_fc, y)   ## dl_dy is columns

            ## Compute Gradients in reverse.
            dl_dx_flatten, dl_dw_fc, dl_db_fc = fc_backward(dl_dy_softmax, y_flatten, w_fc, b_fc, y_fc) 
            dl_dx_maxpool = flattening_backward(dl_dx_flatten, y_maxpool, y_flatten)
            dl_dx_relu = pool2x2_backward(dl_dx_maxpool, y_relu, y_maxpool)
            dl_dx_conv = relu_backward(dl_dx_relu, y_relu, y_maxpool)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dx_conv, x, w_conv, b_conv, y_conv)

            ## Add to the gradients of batch.
            dl_dw_conv_batch += dl_dw_conv
            dl_db_conv_batch += dl_db_conv
            dl_dw_fc_batch += dl_dw_fc
            dl_db_fc_batch += dl_db_fc

        k += 1
        if k == len(mini_batch_x):
            k = 0
        ## Update weights
        w_conv -= (learning_rate/len(mini_batch_x[0]))*dl_dw_conv_batch
        b_conv -= (learning_rate/len(mini_batch_x[0]))*dl_db_conv_batch
        w_fc -= (learning_rate/len(mini_batch_x[0]))*dl_dw_fc_batch
        b_fc -= (learning_rate/len(mini_batch_x[0]))*dl_db_fc_batch   

    return w_conv, b_conv, w_fc, b_fc

if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()