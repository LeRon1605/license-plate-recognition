import os
import cv2 as cv
import numpy as np

folder = './dataset/train/'
chars = [
    '1', '0', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 
    'Z'
]

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

for char in chars:
    path = folder + char
    files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    for file in files:
        image = cv.imread(path + '/' + file)
        ksize = (5, 5)
        image_blur = cv.blur(image, ksize)
        image_rotate_30 = rotate_image(image, 30)
        image_rotate_90 = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        cv.imwrite(path + '/' + os.path.splitext(file)[0] + '_blur.jpg', image_blur)
        cv.imwrite(path + '/' + os.path.splitext(file)[0] + '_rotate_30.jpg', image_rotate_30)
        cv.imwrite(path + '/' + os.path.splitext(file)[0] + '_rotate_90.jpg', image_rotate_90)