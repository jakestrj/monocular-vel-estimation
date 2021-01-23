import pathlib
from pebble import ProcessPool
from pathlib import Path
from tqdm import tqdm

from model import MonocularVelocityNN
from utils import *
from config import config

import numpy as np
import argparse
import os
import glob
import cv2

import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else None

def save_weights(path):
    if not path.exists():
        raise RuntimeError("Cannot Save Model, path not defined")
    torch.save(model.state_dict, str(path / "weights.pt"))

def load_weights(path):
    if not path.exists():
        raise RuntimeError("Cannot Load Model, path not defined")
    model.load_state_dict(torch.load(str(path)))

def train(epoch):
    print(f"running train epoch: {epoch}", flush=True)
    model.train()
    train_loss = 0
    pbar = tqdm(total=len(train_indices))
    getd = lambda i : data[i].to(device, dtype=torch.float)
    for i, data in enumerate(train_loader):
        x1s, x2s, labels = (getd(0), getd(1), getd(2))

        preds = model(x1s, x2s)
        
        loss = nn.MSELoss()(preds, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()

        pbar.update(config["batch_size"])
    
    per_item_loss = train_loss / len(train_loader)

    print(f"mse_res: {per_item_loss}")
    print(f"mse_total: {train_loss}")


@torch.no_grad
def validate(is_labels = True):
    test_loss = 0
    model.eval()
    pbar = tqdm(total=len(val_indices))

    f = open(Path.cwd() / "data/test_pred.txt", 'w')

    test_batch_size = config["test_batch_size"]

    for i, data in enumerate(test_loader):
        x1s, x2s, labels = data[0].to(device, dtype=torch.float),\
                            data[1].to(device, dtype=torch.float),\
                            data[2].to(device, dtype=torch.float)
        test_preds = model(x1s, x2s)
        for p in range(test_batch_size): # batched by 8
            vel = torch.mean(test_preds[p], 0)
            print(f"Frame: {i*test_batch_size + p}, Sec: {(i*test_batch_size + p) / 20}, Vel: {vel}")
            f.write(f"{vel}\n")

        if is_labels:
            test_loss += nn.MSELoss()(test_preds, labels).item()
        pbar.update(test_batch_size)
            
    f.close()
    per_item_loss = test_loss / len(test_loader)
    print(f"mse_res: {per_item_loss}")
    print(f"mse_total: {test_loss}")


def train():
    from preprocess import PreProcess
    p = PreProcess()

    global model, train_indices, val_indices, train_loader, test_loader, optimizer

    model = MonocularVelocityNN(initial_depth=config["depth"])

    dataset = VideoDataLoader(directory=str(Path.cwd() / "data/processed"), delta=config["delta"], 
                                Y=p.labels, depth=config["depth"])

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(1,dataset_size - config["delta"] - config["depth"]))
    split = int(np.floor(config["split"] * dataset_size))
    
    if config["randomize"]:
        np.random.seed(0)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=config["test_batch_size"], sampler=valid_sampler)

    if config["TRAIN"]:
        if not device:
            raise RuntimeError("Only use model with Cuda")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        for epoch in range(1, config["epochs"]+1):
            try:
                train(epoch)
                validate()
            except:
                raise
    
    else:
        load_weights(Path.cwd() / "data/weights.pt")

        if not device:
            raise RuntimeError("Only use model with Cuda")
        model.to(device)

        validate()

def validate_test():
    from preprocess import PreProcess
    p = PreProcess()

    global model, train_indices, val_indices, train_loader, test_loader, optimizer

    model = MonocularVelocityNN(initial_depth=config["depth"])

    dataset = VideoDataLoader(directory=str(Path.cwd() / "data/testprocdir"), delta=config["delta"], 
                                Y=p.labels, depth=config["depth"])

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    val_indices = list(range(1, dataset_size - config["delta"] - config["depth"]))

    # valid_sampler = Sampler(val_indices)

    test_loader = DataLoader(dataset, batch_size=config["test_batch_size"], sampler=val_indices)

    load_weights(Path.cwd() / "data/weights.pt")

    if not device:
        raise RuntimeError("Only use model with Cuda")
    model.to(device)

    validate(is_labels=False)


if __name__ == "__main__":
    # train()
    validate_test()