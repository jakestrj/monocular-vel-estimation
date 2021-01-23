import pathlib
from pebble import ProcessPool
from pathlib import Path
from tqdm import tqdm
from config import config

import numpy as np
import argparse
import os
import glob
import cv2
import torch

class PreProcess():
    def __init__(self):
        self.datadir = Path.cwd() / config["DATADIR"]
        self.rawdir = self.datadir / "raw"
        self.traindir = self.datadir / "processed"
        self.testrawdir = self.datadir / "test"
        self.testprocdir = self.datadir / "testprocdir"

        if not self.rawdir.exists():
            print("Generating frames ..")
            self.rawdir.mkdir(exist_ok=False)
            self.read_images(config["TRAIN_VID"], self.rawdir)
        
        if not self.testrawdir.exists():
            print("Generating frames for test..")
            self.testrawdir.mkdir(exist_ok=False)
            self.read_images(config["TESTS_VID"], self.testrawdir)

        images = list(self.rawdir.rglob('*.jpg'))
        images_test = list(self.testrawdir.rglob('*.jpg'))

        _labels = [float(x) for x in open(self.datadir / config["TRAIN_Y"]).read().splitlines()]
        self.labels = torch.Tensor(_labels)

        if not self.traindir.exists():
            print("Processing images ..")
            self.traindir.mkdir(exist_ok=False)

            pool = ProcessPool(max_workers=config["NUM_THREADS"])
            for _ in tqdm(pool.map(self.transformations, images).result()):
                pass
                
        if not self.testprocdir.exists():
            print("Processing images ..")
            self.testprocdir.mkdir(exist_ok=False)

            pool = ProcessPool(max_workers=config["NUM_THREADS"])
            for _ in tqdm(pool.map(self.transformations, images_test).result()):
                pass

        # self.transformations(images) #can pass pooled chunks into this func

    def read_images(self, video_file, dir):
        os.system(f"ffmpeg -i \"{self.datadir / video_file}\" -q:v 1 \"{dir}/$img%d.jpg\"")

    def gamma_correction(self, img, gamma=1.25):
        inv_gamma = 1.0 / (gamma if gamma > 0 else 0.1)
        table = np.array([((i / 255.0) ** inv_gamma) * 255
		    for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(img, table)

    def sharpen(self, img):
        # https://en.wikipedia.org/wiki/Kernel_(image_processing)
        kernel = np.array([[0, -1, 0],[-1, 5, -1 ],[0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def stretch(self, img, size):
        return cv2.resize(img, (int(img.shape[0] * size), 
                            int(img.shape[1] * size)))


    def transformations(self, img): # batched
        proc = cv2.imread(str(img))

        # gamma correction
        proc = self.gamma_correction(proc)
        
        # sharpen
        proc = self.sharpen(proc)

        # stretch
        proc = self.stretch(proc, size=1.2)

        cv2.imwrite(str(self.traindir if config["TRAIN"] else self.testprocdir / img.name), proc)


if __name__ == "__main__":
    p = PreProcess()