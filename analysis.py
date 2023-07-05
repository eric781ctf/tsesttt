import cv2
import numpy as np
import json
from utils import color_diff, resize
from tqdm import tqdm
import time

class Store:
    def __init__(self, size: tuple[int, int], option):
        self.width = size[0] - size[0] % option.sample
        self.height = size[1] - size[1] % option.sample
        self.union = [[dict() for x in range(self.height // option.sample)]
                      for y in range(self.width // option.sample)]
        self.index = 0
        self.option = option

    def _append(self, pos: tuple[int, int], color: tuple[int, int, int]):
        x = pos[0]
        y = pos[1]

        for c in self.union[x][y]:
            if (color_diff(c, color) < self.option.diff):
                self.union[x][y][c] += 1
                return
        self.union[x][y][color] = 1

    def append(self, frame: np.ndarray):
        self.index += 1

        for y in range(0, self.height, self.option.sample):
            for x in range(0, self.width, self.option.sample):
                self._append(
                    (x // self.option.sample, y // self.option.sample), tuple(frame[x][y]))

    def out(self):
        new_arr = [[dict() for x in range(self.height)]
                   for y in range(self.width)]
        for y in range(0, self.height, self.option.sample):
            for x in range(0, self.width, self.option.sample):
                i = x // self.option.sample
                j = y // self.option.sample
                for color in self.union[i][j]:
                    new_arr[i][j][' '.join(map(str, color))
                                  ] = self.union[i][j][color]
        return new_arr


class Detector:
    def __init__(self, size: tuple[int, int], option):
        self.w = size[0] - size[0] % option.sample
        self.h = size[1] - size[1] % option.sample
        self.option = option
        self.store = Store(size=size, option=option)
        self.frame_count = 0

    def read(self, frame: np.ndarray):
        self.store.append(frame)
        self.frame_count += 1

    def finish(self):
        result = np.zeros(
            (self.w//self.option.sample, self.h//self.option.sample, 3), np.uint8)

        for y in range(0, self.h, self.option.sample):
            for x in range(0, self.w, self.option.sample):
                frequently_color = None
                frequently_count = 0
                i = x // self.option.sample
                j = y // self.option.sample
                for color in self.store.union[i][j]:
                    if (self.store.union[i][j][color] > frequently_count):
                        frequently_color = color
                        frequently_count = self.store.union[i][j][color]
                rate = frequently_count/self.frame_count
                if (frequently_color and rate > self.option.threshold):
                    result[i][j] = (255*rate, 255*rate, 255*rate)

        return cv2.resize(result, (self.h, self.w))


# def detectWithUrl(url, cwd):
#     option = defaultOption
#     option['cwd'] = cwd
#     cap = cv2.VideoCapture(url)
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(url, "start")

#     detector = None

#     index = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if (not detector):
#             detector = Detector(size=tuple(frame.shape[:2]), option=option)
#         if (not (index % option['frame_step'])):
#             detector.read(frame)
#         if (index == length//2):
#             cv2.imwrite(option['cwd']+'frame_shot.png', frame)
#         index += 1

#     if (detector):
#         mat = detector.finish()
#         status = cv2.imwrite(option['cwd']+'f_map.png', mat)
