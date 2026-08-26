import seisbench.data as sbd
import seisbench.util as sbu
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

import os, obspy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def preprocess_data(config, train_dataset, dev_dataset, test_dataset):
    train_generator = sbg.GenericGenerator(train_dataset)
    dev_generator = sbg.GenericGenerator(dev_dataset)
    test_generator = sbg.GenericGenerator(test_dataset)

    train_generator.add_augmentations(config.get_augs())
    dev_generator.add_augmentations(config.get_augs())
    test_generator.add_augmentations(config.get_augs())
    
    return train_generator, dev_generator, test_generator


def train_loop(dataloader, loss_fn, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        x = batch["X"].float().to(model.device)
        pred_raw = model(x)
        args = model.get_model_args()
        args["blinding"] = (0, 0) # Remove blinding that makes leading and trailing samples np.nan
        pred = model.annotate_batch_post(pred_raw, None, args) # In (batch_size, WIN_LEN, channels=3)
        pred = pred.permute(0, 2, 1) # Convert to (batch_size, channels=3, WIN_LEN) 
        loss = loss_fn(pred, batch["y"].to(model.device))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * batch["X"].shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, loss_fn, model):
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            x = batch["X"].float().to(model.device)
            pred_raw = model(x)
            args = model.get_model_args()
            args["blinding"] = (0, 0) # Remove blinding that makes leading and trailing samples np.nan
            pred = model.annotate_batch_post(pred_raw, None, args) # In (batch_size, WIN_LEN, channels=3)
            pred = pred.permute(0, 2, 1) # Convert to (batch_size, channels=3, WIN_LEN)
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n") 

