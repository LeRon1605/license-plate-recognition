import os
import random
import cv2 as cv
import numpy as np

folder = './dataset'
chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 
    'Z'
]

count = 0

for char in chars:
    path = folder + '/final/' + char
    test_path = folder + '/test/' + char
    train_path = folder + '/train/' + char
    files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    random.shuffle(files)
    n_train = int(len(files) * 0.1)
    if n_train < 0:
        n_train = 1
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    for i in range(0, n_train + 1):
        image = cv.imread(path + '/' + files[i], cv.IMREAD_GRAYSCALE)
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        np.reshape(image, (image.shape[0], image.shape[1], 1))
        cv.imwrite(test_path + '/' + str(i) + '.jpg', image)
    for i in range(n_train + 1, len(files)):
        image = cv.imread(path + '/' + files[i], cv.IMREAD_GRAYSCALE)
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        np.reshape(image, (image.shape[0], image.shape[1], 1))
        cv.imwrite(train_path + '/' + str(i - n_train) + '.jpg', image)