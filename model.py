import csv
import os
import sys
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

def loadData(data_path):
    X = []
    y = []
    csv_file = data_path + '/driving_log.csv'
    img_folder = data_path + '/IMG/'
    print('loading csv file %s' % csv_file)
    print('loading images %s' % img_folder)
    with open(csv_file) as f:
        for r in csv.reader(f):
            img_f_c = img_folder + r[0].rsplit('/', 1)[1]
            img_f_l = img_folder + r[1].rsplit('/', 1)[1]
            img_f_r = img_folder + r[2].rsplit('/', 1)[1]
            X.append(cv2.imread(img_f_c)[...,::-1])
            X.append(cv2.imread(img_f_l)[...,::-1])
            X.append(cv2.imread(img_f_r)[...,::-1])
            y_c = float(r[3])
            y.append(y_c)
            y.append(y_c + abs(y_c) * 0.2)
            y.append(y_c - abs(y_c) * 0.2)
            # break
    return np.array(X), np.array(y)

def Model():
    model = Sequential()
    model.add(Lambda(lambda x: (x-128)/128, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(60, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def main(data_path='.'):
    X, y = loadData(data_path)
    print('Data loaded')
    # plt.imshow(X[0])
    # plt.show()
    model = Model()
    print('Starting training')
    model.fit(X, y, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

    model.save('model.h5')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        main()
    else:
        main(sys.argv[1])