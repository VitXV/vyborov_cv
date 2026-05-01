import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from collections import defaultdict

def isRectangle(region):
    return region.extent > 0.95

image = imread('balls_and_rects.png')

hsv = rgb2hsv(image)
h = np.round(hsv[:,:,0],2)

circles = defaultdict(int)
rectangles = defaultdict(int)

for color in np.unique(h)[1:]:
    binary = color == h
    for obj in regionprops(label(binary)):
        if isRectangle(obj):
            rectangles[color] += 1
        else:
            circles[color] += 1

print('Rectangles', sum(rectangles.values()))
for i in rectangles:
    print(str(i).ljust(4,'0'), rectangles[i])
print()
print('Circles', sum(circles.values()))
for i in circles:
    print(str(i).ljust(4,'0'), circles[i])
