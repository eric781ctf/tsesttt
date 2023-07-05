from math import sqrt
import cv2
import numpy as np
from typing import Tuple
import os
import glob

max_v = sqrt(255*255*3)


def color_diff(A: Tuple[int, int, int], B: Tuple[int, int, int]):
    B_1, G_1, R_1 = A
    B_2, G_2, R_2 = B
    rmean = (int(R_1) + int(R_2)) / 2
    R = int(R_1) - int(R_2)
    G = int(G_1) - int(G_2)
    B = int(B_1) - int(B_2)
    return sqrt((2+rmean/256)*(R**2)+4*(G**2)+(2+(255-rmean)/256)*(B**2))/765


def resize(frame, size):
    h = 600
    ratio = size[0]/h
    w = int(size[1]/ratio)
    return cv2.resize(frame, (w, h))


def getSurround(points):
    left, right = min(points, key=lambda p: p[0]), max(
        points, key=lambda p: p[0])
    top, bottom = min(points, key=lambda p: p[1]), max(
        points, key=lambda p: p[1])
    return (left, right, top, bottom)


def getBounding(surround, padding=5):
    left, right, top, bottom = surround
    left[0] -= padding
    right[0] += padding
    top[1] -= padding
    bottom[1] += padding
    r = [(left[0], top[1]), (right[0], top[1]), (right[0], bottom[1]),
         (left[0], bottom[1])]
    box = np.array(r, np.int32)
    return box

def getNameFromPath(path):
    filname = os.path.split(path)[1]
    name, _ = os.path.splitext(filname)
    return name

def crop_Image(img, points):
    rect = cv2.boundingRect(np.array(points))
    
    cropped_img = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    return cropped_img

def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    args_list = glob.glob(f'{search_dir}/**/args.yaml', recursive=True)
    return max(args_list, key=os.path.getctime) if args_list else ''
