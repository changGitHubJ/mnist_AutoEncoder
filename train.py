import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import time

# Parameter
training_epochs = 10001
batch_size = 128
display_step = 1000
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 500
IMG_SIZE = 28

# Batch components
trainingImages = np.zeros((TRAIN_DATA_SIZE, IMG_SIZE*IMG_SIZE + 1))
testImages = np.zeros((TEST_DATA_SIZE, IMG_SIZE*IMG_SIZE))

def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def conv2d_sigmoid(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def conv2dtranspose(input, weight_shape, bias_shape, output_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME'), b))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def inference(x, input_size):
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])
    # Encoding
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [3, 3, 1, 16], [16])
        pool_1 = max_pool(conv_1)
    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [3, 3, 16, 8], [8])
        pool_2 = max_pool(conv_2)
    with tf.variable_scope("conv_3"):
        conv_3 = conv2d(pool_2, [3, 3, 8, 8], [8])
        pool_3 = max_pool(conv_3)
    
    # Decoding
    with tf.variable_scope("conv_4"):
        conv_4 = conv2dtranspose(pool_3, [3, 3, 8, 8], [8], [input_size, 7, 7, 8])
    with tf.variable_scope("conv_5"):
        conv_5 = conv2dtranspose(conv_4, [3, 3, 8, 8], [8], [input_size, 14, 14, 8])
    with tf.variable_scope("conv_6"):
        conv_6 = conv2dtranspose(conv_5, [3, 3, 16, 8], [16], [input_size, 28, 28, 16])
    with tf.variable_scope("conv_7"):
        conv_7 = conv2d_sigmoid(conv_6, [3, 3, 16, 1], [1])
    
    decoded = tf.reshape(conv_7, [-1, IMG_SIZE * IMG_SIZE])
    return decoded

def loss(x, decoded):
    cross_entropy = -1. *x *tf.log(decoded) - (1. - x) *tf.log(1. - decoded)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def training(cost):
    train_op = tf.train.AdagradOptimizer(0.1).minimize(cost)
    return train_op

def openfile(filename):
    file = open(filename)
    VAL = []
    while True:
        line = file.readline()
        if(not line):
            break
        val = line.split(' ')
        VAL.append(val)
    return VAL

def read_training_data():
    fileImg = open('./data/trainImage.txt', 'r')
    for i in range(TRAIN_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        trainingImages[i, :] = val[0:IMG_SIZE*IMG_SIZE + 1]
    for i in range(TRAIN_DATA_SIZE):
        for j in range(1,IMG_SIZE*IMG_SIZE + 1):
            trainingImages[i, j] /= 255.0

def defineBatchComtents():
    num = np.linspace(0, TRAIN_DATA_SIZE - 1, TRAIN_DATA_SIZE)
    num = num.tolist()
    component = random.sample(num, batch_size)
    return component

def next_batch(batch_component):
    num = sorted(batch_component)
    lineNum = 0
    cnt = 0
    batch_x = []
    while True:
        if(cnt == batch_size):
            break
        else:
            if(int(num[cnt]) == int(trainingImages[lineNum, 0])):
                image = trainingImages[lineNum, 1:IMG_SIZE*IMG_SIZE + 1]
                batch_x.append(image)
                cnt += 1
        lineNum += 1

    return np.array(batch_x)

def read_test_data():
    fileImg = open('./data/testImage.txt', 'r')
    for i in range(TEST_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        testImages[i, :] = val[1:IMG_SIZE*IMG_SIZE + 1]
    for i in range(TEST_DATA_SIZE):
        for j in range(IMG_SIZE*IMG_SIZE):
            testImages[i, j] /= 255.0

if __name__=='__main__':    
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            x = tf.placeholder("float", [None, IMG_SIZE*IMG_SIZE])
            input_size = tf.placeholder(tf.int32)
            
            read_training_data()
            
            output = inference(x, input_size)
            cost = loss(x, output)
            train_op = training(cost)
            sess = tf.Session()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Training cycle
            for epoch in range(training_epochs):
                batch_component = defineBatchComtents()
                minibatch_x = next_batch(batch_component)
                sess.run(train_op, feed_dict={x: minibatch_x, input_size: batch_size})

                # display logs per step
                if epoch % display_step == 0:
                    train_loss = sess.run(cost, feed_dict={x: minibatch_x, input_size: batch_size})
                    print('  step, loss = %6d: %6.3f' % (epoch, train_loss))
                    
            print("Optimizer finished!")

            # generate decoded image with test data
            read_test_data()
            decoded_imgs = sess.run(output, feed_dict={x: testImages, input_size: TEST_DATA_SIZE})
            result_loss = sess.run(cost, feed_dict={x: testImages, input_size: TEST_DATA_SIZE})
            print('loss (test) = ', result_loss)

            n = 10
            for i in range(n):
                # display original
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(testImages[i].reshape(IMG_SIZE, IMG_SIZE))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display deconded image
                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(decoded_imgs[i].reshape(IMG_SIZE, IMG_SIZE))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            #plt.show()
            plt.savefig('result.png')
