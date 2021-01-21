import pathlib
from pebble import ProcessPool
from pathlib import Path
from tqdm import tqdm

import numpy as np
import argparse
import os
import glob
import cv2
import torch

# parser = argparse.ArgumentParser()
# parser.add_argument("--datadir", type=Path, default="data")
# parser.add_argument("--num-threads", type=int, default=4, help="# of threads to pool")

# args = parser.parse_args()

DATADIR = "data"
TRAIN_VID = "train.mp4"
TRAIN_Y = "train.txt"
TEST_VID = "test.mp4"
REPROCESS = False
NUM_THREADS = 4

class PreProcess():
    def __init__(self):
        self.datadir = Path.cwd() / DATADIR
        self.rawdir = self.datadir / "raw"
        self.traindir = self.datadir / "processed"

        if not self.rawdir.exists():
            print("Generating frames ..")
            self.rawdir.mkdir(exist_ok=False)
            self.read_images(TRAIN_VID)

        images = list(self.rawdir.rglob('*.jpg'))

        labels = [float(x) for x in open(self.datadir / TRAIN_Y).read().splitlines()]
        labels = torch.Tensor(labels)

        if not self.traindir.exists():
            print("Processing images ..")
            self.traindir.mkdir(exist_ok=False)

            pool = ProcessPool(max_workers=NUM_THREADS)
            for _ in tqdm(pool.map(self.transformations, images).result()):
                pass

        # self.transformations(images) #can pass pooled chunks into this func

    def read_images(self, video_file : str):
        os.system(f"ffmpeg -i \"{self.datadir / video_file}\" -q:v 1 \"{self.rawdir}/$img%d.jpg\"")

    def gamma_correction(self, img, gamma=1.25):
        inv_gamma = 1.0 / (gamma if gamma > 0 else 0.1)
        table = np.array([((i / 255.0) ** inv_gamma) * 255
		    for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(img, table)

    def sharpen(self, img):
        # https://en.wikipedia.org/wiki/Kernel_(image_processing)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(img, -1, kernel)

    def transformations(self, img): # batched
        # for img in tqdm(images):
        proc = cv2.imread(str(img))

        # gamma correction
        proc = self.gamma_correction(proc)

        # sharpen
        proc = self.sharpen(proc)

        cv2.imwrite(str(self.traindir / img.name), proc)



p = PreProcess()