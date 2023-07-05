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

datetime_ = datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser()

parser.add_argument('path', nargs='?', default=None, help='Path of target directory')

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
parser.add_argument('--resume', nargs='?', const=True, default=False,
                    help='Resume most recent processing')
parser.add_argument("-l", "--limit", type=int, default=None,
                    help="Limit the number of input videos")


if __name__ == "__main__":
    args = parser.parse_args()
    #從中斷點繼續執行
    if args.resume:
        args_path = get_latest_run()
        with open(args_path) as f:
            args = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        args.resume = True
        #print(args)
    else:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'args.yaml'), 'w', encoding = 'big5') as f:
            yaml.dump(vars(args), f, sort_keys=False)

    # 設定日誌級別為 DEBUG
    logging.basicConfig(filename=os.path.join(args.output, 'debug.log'), level=logging.DEBUG)
    
    path = args.path

    files = []  # 儲存檔案名稱的list
    before_cutting = 0
    after_cutting = 0
    template = 0
    total_duration = 0
    corruption = 0
    result_path = os.path.join(args.output, 'result.csv')
    
    for filename in os.listdir(path):  # 遍歷目錄中的所有檔案
            if os.path.isfile(os.path.join(path, filename)) and filename.endswith('.mp4'):  #確保檔案類型皆為mp4
                files.append(filename)  # 將檔案名稱加入list

    try:
        if (args.resume == False) and (args.limit is None):
            samples = files
        elif (args.resume == False) and (args.limit is not None):
            #隨機抽樣影片
            index = random.sample(range(len(files)), args.limit)
            samples = [files[id] for id in index]
        elif (args.resume == True) and (args.limit is None):
            if os.path.isfile(result_path):
                df = pd.read_csv(result_path, encoding = 'big5')
                if not df.empty:
                    processed = list(df['parent_doc_id'])
                    samples = list(filter(lambda x: x not in processed, files))
                else:
                    samples = files
            else:
                samples = files
        else:
            raise Exception           
    except Exception as e:
        print(e)
        print('Can\'t resume last run, because args.limit is not None')
        logging.warning('Can\'t resume last run, because args.limit is not None')
        sys.exit(0)
    
    
    
    #確認影片是否毀損
    for filename in tqdm(samples, desc = 'Verifying videos integrity', colour='yellow'):       
        filepath = os.path.join(path, filename)
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise cv2.error
        except cv2.error:
            logging.warning(f'Ignoring video {filename} due to file corruption')
            samples.remove(filename)
            if (args.limit is not None) and (args.limit != len(files)):
                resample_index = random.choice(range(len(files)))
                while resample_index in index:
                    resample_index = random.choice(range(len(files)))
                samples.append(files[resample_index])
        except Exception:
            print(f'May have some problems when varifying {filename}')
            logging.warning(f'May have some problems when varifying {filename}')
            sys.exit(0)
            

    
    for filename in tqdm(samples, desc = 'Processing videos', colour="green"): #Cutting out of the template
        #偵測影片模板
        try:
            t0 = time.time()
            vidpath = os.path.join(path, filename)
            args.video = vidpath
            output_path = os.path.join(args.output, filename)
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

        #重新編碼影片與加入音訊
        try:
            after_cutting = EncodeVideo(output_dir, path)
            t2 = time.time()
        except FileNotFoundError as debug_info:
            print(f'Target video not found while encoding video {filename}')
            logging.error(f'Target video not found while encoding video {filename}')
            logging.debug(debug_info)
            sys.exit(0)
        except Exception as debug_info:
            print(f'Program terminated because Unknown error happened while encoding video {filename}')
            logging.error(f'Program terminated because Unknown error happened while encoding video {filename}')
            logging.debug(debug_info)
            sys.exit(0)

        #紀錄相關數據
        try:
            labels = ['parent_doc_id', 'time_start', 'time_end', 'content', 'content_boundary [xywh]', 'storage_usage [%]',
                       'detect_template [processing time]', 'reEncode_video [processing time]', 'data_path', 'processed_at']
            
            item = {'parent_doc_id': filename, 'time_start': '00:00:00', 'time_end': str(timedelta(seconds = round(duration))),
                     'content': '', 'content_boundary [xywh]': boundary, 'storage_usage [%]': round((after_cutting / before_cutting) * 100, 2),'detect_template [processing time]': str(timedelta(seconds = round(t1 - t0))),
                       'reEncode_video [processing time]': str(timedelta(seconds = round(t2 - t1))), 'data_path': output_dir, 'processed_at': datetime.now().strftime("%Y/%m/%d %H:%M:%S")}
            
            with open(result_path, 'a', encoding='big5', newline = '') as f:
                writer = csv.DictWriter(f, fieldnames = labels)
                if not os.path.isfile(result_path) or f.tell() == 0:
                    writer.writeheader()
                writer.writerow(item)
        except Exception as e:
            print(e)
            print('Program terminated while writting result')
            logging.error('Program terminated while writting result')
            sys.exit(0)
            
    
    # reduction_percentage = (after_cutting / before_cutting) * 100
    # before_cutting = humanize.naturalsize(before_cutting)
    # after_cutting = humanize.naturalsize(after_cutting)
    # total_duration = str(timedelta(seconds = round(total_duration)))
         
    # with open(os.path.join(args.output, 'result.txt'), 'w') as f:
    #     print(f'執行結束時間：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', file = f)

    #     print(f'處理 {len(samples)} 部影片耗時: {str(timedelta(seconds = round(t2 - t0)))}, '
    #           f'影片總時長: {total_duration}, '
    #           f'去除模板耗時: {str(timedelta(seconds = round(t1 - t0)))}, '
    #           f'轉換至h264編碼耗時: {str(timedelta(seconds = round(t2 - t1)))}', file = f)
        
    #     print(f'去除模板前使用儲存空間: {before_cutting}, '
    #           f'去除模板後使用儲存空間: {after_cutting}, '
    #           f'降低使用空間百分比: {reduction_percentage:.2f}%', file = f)
        
    #     print(f'具有模板的影片數量: {template}, '
    #           f'所占百分比: {((template / len(samples)) * 100):.2f}% \n', file = f)
        
