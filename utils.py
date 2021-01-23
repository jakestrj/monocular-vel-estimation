import os
import cv2
import glob
import torch
from torch.utils.data import Dataset, DataLoader

class VideoDataLoader(Dataset):

    def __init__(self, depth, labels, dir, frame_delta= 1):
        self.depth = depth
        self.labels = labels
        self.dir = dir
        self.frame_delta = frame_delta

    def __len__(self):
        return len(glob.glob(f"{self.dir}/*.jpg"))

    def _img(self, index):
        x = cv2.imread(f"{self.dir}/{index}.jpg")
        return torch.Tensor(x)

    def getimages(self, index):
        lay1 = list()
        lay2 = list()
        for i in range(index, index + self.depth):
            lay1.append(self._img(index).T)
            lay2.append(self._img(index + self.frame_delta).T)
        
        lay1 = torch.stack([x for x in lay1], dim=0)
        lay2 = torch.stack([x for x in lay2], dim=0) # along third dim axiss
    
        return lay1, lay2

    def __getitem__(self, index):
        _x1, _x2 = self.getimages(index)

        block1 = self.labels[index : index + self.depth]
        block2 = self.labels[index + self.frame_delta : \
            index + (self.frame_delta + self.depth)]
        avg_labels = torch.mean( \
            torch.stack([block1, block2], dim=0), dim=0)
        return block1, block2, avg_labels