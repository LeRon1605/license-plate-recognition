import cv2
import numpy as np
from imutils import contours
from keras.models import load_model
import keras.utils as image
from yolo_prediction import YOLO_Pred

classes = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K',
    19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T', 26: 'U', 27: 'V',
    28: 'X', 29: 'Y', 30: 'Z'
}

def take_second(s):
    return s[1]

def segment_char(license_plate):
    license_plate = cv2.resize(license_plate, (500, 400))
    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow('Thresh', thresh)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method = "top-to-bottom")

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([x, y, w, h])

    min_y = None
    first_line = []
    second_line = []

    for box in boxes:
        x, y, w, h = box
        char_area = license_plate[y:y + h, x:x + w]
        char_area = cv2.resize(char_area, (12, 28), interpolation=cv2.INTER_AREA)
        char_area = np.expand_dims(char_area, axis = 0)
        char_area = char_area / 255
        predictions = model.predict(char_area)
        max_value = np.max(predictions, axis = 1)
        index = np.argmax(predictions, axis = 1)
        char = classes[index[0]]
        # print(max_value)
        if max_value < 0.8:
            continue

        if min_y == None or min_y > y:
            min_y = y
        if y - min_y > (h / 2):
            second_line.append((char, x))
        else:
            first_line.append((char, x))

        cv2.rectangle(license_plate, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.putText(license_plate, char, (x + w, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

    first_line = sorted(first_line, key = take_second)
    second_line = sorted(second_line, key = take_second)
    if len(second_line) == 0:
        text = "".join([str(ele[0]) for ele in first_line])
    else: 
        text = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])
    cv2.putText(license_plate, text, (int(3 * license_plate.shape[1] / 5), license_plate.shape[0] - 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=(0, 0, 255), thickness=2)
    return license_plate

license_plate_model = YOLO_Pred('./model/weights_2/best.onnx')
model = load_model('./model/model_2.h5')

image = cv2.imread('./images/6.jpg')
# image = cv2.resize(image, (640, 640))

licenses = license_plate_model.predict(image)

for (index, license_plate) in enumerate(licenses):
    x, y, w, h = license_plate
    license_plate = image[y:y + h, x:x + w]
    result = segment_char(license_plate)
    cv2.imshow('License ' + str(index), result)

cv2.imshow('Image', image)
cv2.waitKey()