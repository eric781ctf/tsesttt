# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:55:02 2023

@author: Iorg-DE
"""

from procedure import getStaicMap, process, roiVideo, EncodeVideo
from utils import get_latest_run
import os
import sys
import argparse
from tqdm import tqdm
import time
from datetime import timedelta,  datetime
import humanize
import logging
import cv2
import random
import yaml
import pandas as pd
import csv
import yaml
import threading

datetime_ = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(f'runs/{datetime_}/', exist_ok=True)
logging.basicConfig(filename=os.path.join(f'runs/{datetime_}/', 'debug.log'), level=logging.DEBUG)

path = 'C:/Users/Iorg-DE/Desktop/Douyin/20220717 douyin/video/'
files = [x for x in os.listdir(path) if x.split('.')[1] == 'mp4']
#%% 檢查影片是否有問題
class check_audio(threading.Thread):
       def __init__(self, num):
           threading.Thread.__init__(self)
           self.num = num
                
       def run(self):
           while len(files)>0:
               file = files.pop(0)
               filepath = path + file
               try:
                   cap = cv2.VideoCapture(filepath)
                   if not cap.isOpened():
                       raise cv2.error
                   finish_check_files.append(file)
               except cv2.error:
                   logging.warning(f'Ignoring video {filename} due to file corruption')
                   files.remove(filename)
               except Exception:
                   print(f'May have some problems when varifying {filename}')
                   logging.warning(f'May have some problems when varifying {filename}')
                   sys.exit(0)

finish_check_files = []
threads = []
start = time.time()
for i in range(20):
    threads.append(check_audio(i))
    threads[i].start()

for i in range(20):
    threads[i].join()
print(time.time()-start)

#%%
class detect_template(threading.Thread):
       def __init__(self, num):
           threading.Thread.__init__(self)
           self.num = num
                
       def run(self):
           while len(finish_check_files)>0:
               filename = finish_check_files.pop(0)
               parser = argparse.ArgumentParser()
               parser.add_argument('video', nargs='?', default=None, help='Path to video')
               parser.add_argument("-d", "--diff", type=int, default=0.08,
                                   help="Threshold to define if the pixel changed")
               parser.add_argument("-t", "--threshold", type=int, default=0.85,
                                   help="Threshold to determine if the pixel is static")
               parser.add_argument("-s", "--step", type=int, default=10,
                                   help="Scan the video by a stepped frame count")
               parser.add_argument("-p", "--sample", type=int, default=5,
                                   help="Space between sampling pixels")
               parser.add_argument("-o", "--output", type=str, default=f'runs/{datetime_}/',
                                   help="Output directory")
               args = parser.parse_args()
               #偵測影片模板
               try:
                   t0 = time.time()
                   vidpath = os.path.join(path, filename)
                   args.video = vidpath
                   output_path = os.path.join('runs/datetime_', filename)
                   before_cutting = os.path.getsize(vidpath)
                   static, _ = getStaicMap(args)
                   edge, recheck, found, roi = process(static)
                   if found:
                       output_dir = os.path.join(args.output,f'videos/with_template/{filename}/')
                       os.makedirs(output_dir, exist_ok=True)
                   else:
                       output_dir = os.path.join(args.output, f'videos/without_template/{edge}/{filename}/')
                       os.makedirs(output_dir, exist_ok=True)

                   if recheck:
                       logging.debug(f'video {filename}\'s content contour = 0, Inspection required!')

                   cv2.imwrite(os.path.join(output_dir, 'static.png'), static)
                   boundary, duration = roiVideo(vidpath, roi, output_dir)
                   t1 = time.time()
                   print('Finish 1')
               except Exception as debug_info:
                   print(f'Program terminated while detecting a template from {filename}')
                   logging.error(f'Program terminated while detecting a template from {filename}')
                   logging.debug(debug_info)
                   sys.exit(0)
threads = []
before_cutting = 0
after_cutting = 0
total_duration = 0
corruption = 0
start = time.time()
for i in range(8):
    threads.append(detect_template(i))
    threads[i].start()

for i in range(8):
    threads[i].join()
print(time.time()-start)


#%%
start = time.time()
before_cutting = 0
after_cutting = 0
template = 0
total_duration = 0
corruption = 0
for filename in tqdm(finish_check_files, desc = 'Processing videos', colour="green"): #Cutting out of the template
    parser = argparse.ArgumentParser()
    parser.add_argument('video', nargs='?', default=None, help='Path to video')
    parser.add_argument("-d", "--diff", type=int, default=0.08,
                        help="Threshold to define if the pixel changed")
    parser.add_argument("-t", "--threshold", type=int, default=0.85,
                        help="Threshold to determine if the pixel is static")
    parser.add_argument("-s", "--step", type=int, default=10,
                        help="Scan the video by a stepped frame count")
    parser.add_argument("-p", "--sample", type=int, default=5,
                        help="Space between sampling pixels")
    parser.add_argument("-o", "--output", type=str, default=f'runs/{datetime_}/',
                        help="Output directory")
    args = parser.parse_args()
    #偵測影片模板
    try:
        t0 = time.time()
        vidpath = os.path.join(path, filename)
        args.video = vidpath
        output_path = os.path.join('runs/datetime_', filename)
        before_cutting = os.path.getsize(vidpath)
        static, _ = getStaicMap(args)
        edge, recheck, found, roi = process(static)
        if found:
            template += 1
            output_dir = os.path.join(args.output,f'videos/with_template/{filename}/')
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.join(args.output, f'videos/without_template/{edge}/{filename}/')
            os.makedirs(output_dir, exist_ok=True)

        if recheck:
            logging.debug(f'video {filename}\'s content contour = 0, Inspection required!')

        cv2.imwrite(os.path.join(output_dir, 'static.png'), static)
        boundary, duration = roiVideo(vidpath, roi, output_dir)
        t1 = time.time()
    except Exception as debug_info:
        print(f'Program terminated while detecting a template from {filename}')
        logging.error(f'Program terminated while detecting a template from {filename}')
        logging.debug(debug_info)
        sys.exit(0)

print(time.time()-start)























