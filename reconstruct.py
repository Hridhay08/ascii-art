import numpy as np
import cv2
import math

def inv_stretch_uint8(img, a, c):
    arr = img.astype(float)
    exp_term = -4*math.tan(a*1.57/100)/255
    return (np.log((255/arr)-1)/exp_term + c).astype(np.uint8)

art = open('art4.txt', 'r')
lines = art.readlines()
rows = len(lines)
cols = max([len(line.strip()) for line in lines])

chars = """ .'`^",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"""
alph_len = len(chars)

font_size = 6
step_row = 11
step_col = 4

img = np.zeros((rows*step_row, cols*step_col), dtype=np.uint8)

for i, line in enumerate(lines):
    line = line.strip().rjust(cols, " ")
    for j, char in enumerate(line):
        index = chars.rfind(char)
        lum = (index/(alph_len-1))*255 if alph_len>1 else 0
        img[i*step_row:(i+1)*step_row, j*step_col:(j+1)*step_col] = lum

img = inv_stretch_uint8(img, 85, 128)
cv2.imshow('Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
