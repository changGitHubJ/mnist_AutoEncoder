import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from PIL import Image

TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 500
IMG_SIZE = 28

if __name__ == "__main__":

    init = tf.global_variables_initializer()
    sess = tf.Session()
    with sess.as_default():

        if not os.path.exists('./data'):
            os.mkdir('./data')

        # remove old file
        if(os.path.exists('./data/trainImage.txt')):
           os.remove('./data/trainImage.txt')
        if(os.path.exists('./data/testImage.txt')):
           os.remove('./data/testImage.txt')
        
        # load training images (1-10000)        
        for k in range(TRAIN_DATA_SIZE):
            filename = './images/training/image_' + str(k + 1) + '.jpg'
            print(filename)
            imgtf = tf.read_file(filename)
            img = tf.image.decode_jpeg(imgtf, channels=1)
            array = img.eval()
            line = str(k)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    line = line + ',' + str(array[i, j, 0])
            line = line + '\n'
            file = open('./data/trainImage.txt', 'a')
            file.write(line)
            file.close()

        # load validation images (1-1000)
        evaluate_label = np.loadtxt('./images/evaluation/label.txt')
        for k in range(TEST_DATA_SIZE):
            filename = './images/evaluation/image_' + str(k + 1) + '.jpg'
            print(filename)
            imgtf = tf.read_file(filename)
            img = tf.image.decode_jpeg(imgtf, channels=1)
            array = img.eval()
            line = str(k)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    line = line + ',' + str(array[i, j, 0])
            line = line + '\n'
            file = open('./data/testImage.txt', 'a')
            file.write(line)
            file.close()

        sess.close()
