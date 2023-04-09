import cv2
import numpy as np
import os

chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 
    'Z'
]

image_path = './dataset/data_from_nang/final/'
dest_path = './dataset/final/'

for char in chars:
    path = image_path + char
    files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    for file in files:
        image = cv2.imread(path + '/' + file)
        image = cv2.resize(image, (28, 12))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dest_path + char + '/' + file, image)