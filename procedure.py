import sys
import cv2
import numpy as np
import os
from utils import getSurround, getNameFromPath
from analysis import Detector
from tqdm import tqdm
import numpy as np
import subprocess
os.environ['FFPROBE_PATH']='C:/Users/IORG-DE/.conda/envs/check_dsc/Lib/site-packages/ffprobe'

def getStaicMap(args):
    path = args.video
    #entity_name = getNameFromPath(path)
    #output_path = os.path.join('./tmp', entity_name)
    #os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(path)
    result = detect(cap, args)
    cap.release()

    #cv2.imwrite(os.path.join(output_path, 'static.jpg'), result['static'])
    #cv2.imwrite(os.path.join(output_path, 'sample.jpg'), result['sample'])
    return result['static'], result['sample']

def detect(cap, args):
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = 0
    result = dict()
    
    ret, frame = cap.read()
    detector = Detector(size=tuple(frame.shape[:2]), option=args)

    with tqdm(total=length, desc="Collecting pixel info", position=1) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if (not (index % args.step)):
                detector.read(frame)
            if (index == length//2):
                result['sample'] = frame.copy()
            index += 1
            pbar.update(1)

    if (detector):
        mat = detector.finish()
        result['static'] = mat.copy()

    return result

def roiVideo(vidpath, roi, output_dir):
    
    
    cap = cv2.VideoCapture(vidpath)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = length / fps

    rect = cv2.boundingRect(np.array(roi))
   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    content = cv2.VideoWriter(os.path.join(output_dir,'content.mp4'), fourcc, fps, (rect[2], rect[3]))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    template = cv2.VideoWriter(os.path.join(output_dir,'template.mp4'), fourcc, fps, (width, height))

    with tqdm(total=length, desc="Cropping", position=1) as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                content.write(frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
                frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 0
                template.write(frame)
            else:
                break
            pbar.update(1)

    cap.release()
    content.release()
    template.release()

    return rect, duration

def process(static_map):
    f_map = cv2.cvtColor(static_map, cv2.COLOR_BGR2GRAY)
    
    full_w = f_map.shape[1]
    full_h = f_map.shape[0]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 11))

    d_map = cv2.erode(f_map, kernel)  # dynamic
    d_map = cv2.dilate(d_map, kernel)
    d_map = 255 - d_map
    d_map = cv2.threshold(
        d_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  
    cnts, hierarchy = cv2.findContours(
        d_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    

    d_area = 0
    d_cnt = None
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if (area > d_area):
            d_area = area
            d_cnt = cnt
    if d_cnt is not None:
        # with open(f'special/sampling_result.txt', 'w') as f:
        #     for i in range(d_cnt.shape[0]):
        #         print(d_cnt[i][0], file = f)
        approx = cv2.approxPolyDP(d_cnt, 0.01*cv2.arcLength(d_cnt, True), True)
        print("Dynamic area got", len(approx), "edges")
        if (len(approx) == 4):
            print("Template found!")
            found = True
        else:
            print("It might not have a decisive template!")  
            found = False 
        recheck = False
        edges = len(approx)
    else:
        #完全找不到物件輪廓時，將輪廓設定為整部影片的邊界，並將有這種狀況的影片檔名紀錄至debug.log以便重新檢查
        d_cnt = np.array([[[0, 0]],[[full_w,0]], [[0, full_h]], [[full_w, full_h]]])
        approx = cv2.approxPolyDP(d_cnt, 0.01*cv2.arcLength(d_cnt, True), True)
        print("It might not have a decisive template!")  
        found = False 
        recheck = True
        edges = 0

    return edges, recheck, found, (np.squeeze(approx))

def EncodeVideo(root_dir, oriVideo_dir):
    filesize = 0
    for root, dirs, files in tqdm(os.walk(root_dir), desc='Traversing all subdirectories in the target directory'):
        for file in files:
            if file.endswith('.mp4'):
                if root.split('/')[-1] == '':
                    oriVideo_name = root.split('/')[-2]
                else:
                    oriVideo_name = root.split('/')[-1]
                oriVideo_path = os.path.join(oriVideo_dir, oriVideo_name)
                intput_path = os.path.join(root, file)
                output_path = os.path.join(root, 'ff_' + file)
                command = f'ffprobe -v 0 -select_streams v:0 -show_entries stream=bit_rate -of compact=p=0:nk=1 \'{oriVideo_path}\''
                bitrate = subprocess.check_output(command, shell = True).decode('utf-8').strip()
                command = f'ffmpeg -i \'{intput_path}\' -i \'{oriVideo_path}\' -c:v libx264 -b:v {bitrate} -colorspace bt709 -color_primaries bt709 -color_trc bt709 -pix_fmt yuv420p -map 0:v:0 -map 1:a:0 \'{output_path}\''
                os.system(command)
                filesize += os.path.getsize(output_path)
                os.replace(output_path, intput_path)
    return filesize
