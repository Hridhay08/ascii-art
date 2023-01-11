import cv2
import os
import numpy as np
import math
import mediapipe as mp

def stretch_uint8(img, a, c):
    arr = img.astype(float)
    exp_term = -4*math.tan(a*1.57/100)/255
    return (255/(1+np.exp(exp_term*(arr-c)))).astype(np.uint8)

class SelfieSegmentation():

    def __init__(self, model=1):
        self.model = model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(self.model)

    def removeBG(self, img, imgBg=(255, 255, 255), threshold=0.1):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.selfieSegmentation.process(imgRGB)
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > threshold
        if isinstance(imgBg, tuple):
            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            _imgBg[:] = imgBg
            imgOut = np.where(condition, img, _imgBg)
        else:
            imgOut = np.where(condition, img, imgBg)
        return imgOut


chars = """ .'`^",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"""
alph_len = len(chars)
# font_size = 14
# step_row = 25
# step_col = 11
font_size = 6
step_row = 11
step_col = 4

kernel = np.ones((7,7),np.uint8)
segmentor = SelfieSegmentation()
os.system('cls')
img = cv2.imread('img.jpg')
img = cv2.flip(img, 1)
#img = segmentor.removeBG(img, (0,0,0), threshold = 0.8)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (0, 0), fx = 1.5, fy = 1.5)
img = stretch_uint8(img, 87, 177)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
rows = (img.shape[0]//step_row)*step_row
cols = (img.shape[1]//step_col)*step_col
img = img[:rows, :cols]
for i in range(0, rows, step_row):
    for j in range(0, cols, step_col):
        block = img[i:i+step_row, j:j+step_col]
        val = np.average(block)
        index = int((val/255)*(alph_len-1))
        print(chars[index], end='')
    print('')
cv2.imshow('Window', img)
cv2.waitKey(0)
