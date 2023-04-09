import cv2
import os
from glob import glob
from xml.etree import ElementTree as et 

images_path = './dataset/data_from_nang/data/'
final_path = './dataset/final/'

xml_path = './dataset/data_from_nang/anotations'

chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 
    'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 
    'Z'
]

for char in chars:
    path = './dataset/data_from_nang/final/' + char
    if not os.path.exists(path):
        os.makedirs(path)

def processing(xml):
    tree = et.parse(xml)
    root = tree.getroot()

    image_path = images_path + root.find('filename').text
    image = cv2.imread(image_path)

    objs = root.findall('object')
    for obj in objs:
        name = obj.find('name').text

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)

        crop_image = image[ymin:ymax, xmin:xmax]
        cv2.imwrite(final_path + name + '/' + root.find('filename').text, crop_image)


xmlfiles = glob('./dataset/data_from_nang/anotations/*.xml')
replace_text = lambda x: x.replace('\\','/')
xmlfiles = list(map(replace_text, xmlfiles))

for xml in xmlfiles:
    processing(xml)