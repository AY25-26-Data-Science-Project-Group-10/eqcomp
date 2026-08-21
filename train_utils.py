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

from sklearn.metrics import precision_score, recall_score, f1_score

EQDET_DIR = Path("../eqdet")
EQDET_DATA_DIR = EQDET_DIR / "data"

QKML_EQ_DIR = EQDET_DATA_DIR / "qkml_earthquakes"
QKML_EX_DIR = EQDET_DATA_DIR / "qkml_explosions"

WAVEFORM_EQ_DIR = EQDET_DATA_DIR / "waveforms_earthquakes_nonoise"
WAVEFORM_EX_DIR = EQDET_DATA_DIR / "waveforms_explosions_nonoise"
WAVEFORM_NOISE_DIR = EQDET_DATA_DIR / "waveforms_noise_only"

FILENAME_WINDOWS = "windows.csv"
FILENAME_EQ_Y = "eq_labels.csv"
FILENAME_EX_Y = "ex_labels.csv"
FILENAME_NOISE_Y = "noise_labels.csv"

DIR_METADATA = Path(os.getcwd())
FILENAME_METADATA = "metadata.csv"
FILENAME_WAVEFORMS = "waveforms.hdf5"

SAMPLING_RATE = 100
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.10


def get_true_pick(sample):
    return int(np.argmax(sample["y"]))  # true pick time from probability trace

def get_predicted_pick(prob_trace, threshold=0.1):
    peak_idx = int(np.argmax(prob_trace))
    peak_val = prob_trace[peak_idx]
    return peak_idx if peak_val > threshold else None

def evaluate_model(model, test_generator, test, model_name, mae_thresholds=[0.2, 0.5, 1.0]):
    results = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(test_generator))):
            sample = test_generator[i]
            meta = test.get_sample(i)[1]

            # Ground truth event type
            event_type = meta["event_type"]

            # -----------------------------
            # 1. Forward pass
            # -----------------------------
            x = torch.tensor(sample["X"]).float().to(model.device).unsqueeze(0)
            raw_pred = model(x)

            # -----------------------------
            # 2. Standardise output format using annotate_batch_post
            # -----------------------------
            args = model.get_model_args()
            args["blinding"] = (0, 0) # Remove blinding that makes leading and trailing samples np.nan
            pred = model.annotate_batch_post(raw_pred, None, args)
            pred = pred.permute(0, 2, 1)   # (B, C, T)
            pred = pred.cpu().numpy()

            # -----------------------------
            # 3. Extract P and S probability traces
            # -----------------------------
            C = len(model.labels)

            if C == 3:
                # EQTransformer / PhaseNet
                prob_P = pred[0, 1, :]
                prob_S = pred[0, 2, :]
            elif C == 1:
                # EQCCT: single phase channel → treat as S
                prob_P = pred[0, 0, :] if "P" in model.labels else None
                prob_S = pred[0, 0, :] if "S" in model.labels else None
            else:
                raise ValueError(f"Unexpected channel count: {C}")

            # -----------------------------
            # 4. Ground truth P/S picks
            # -----------------------------
            true_P = meta.get("trace_p_arrival_sample", None) if event_type != "noise" else None
            true_S = meta.get("trace_s_arrival_sample", None) if event_type != "noise" else None

            # -----------------------------
            # 5. Predict P/S picks
            # -----------------------------
            pred_P = get_predicted_pick(prob_P) if prob_P is not None else None
            pred_S = get_predicted_pick(prob_S) if prob_S is not None else None

            # -----------------------------
            # 6. Evaluate P and S separately
            # -----------------------------
            for phase_name, true_t, pred_t in [
                ("P", true_P, pred_P),
                ("S", true_S, pred_S)
            ]:
                for th in mae_thresholds:
                    if event_type == "noise":
                        y_true = 0
                        y_pred = 1 if pred_t is not None else 0
                        error = None
                    else:
                        y_true = 1
                        if pred_t is None:
                            y_pred = 0
                            error = None
                        else:
                            error = abs(pred_t - true_t) / SAMPLING_RATE
                            y_pred = 1 if error <= th else 0

                    results.append({
                        "Model": model_name,
                        "Event Type": event_type,
                        "Phase": phase_name,
                        "Threshold": th,
                        "Predicted": y_pred,
                        "True": y_true,
                        "Error": error
                    })

    return results


def aggregate_metrics(results):
    df = pd.DataFrame(results)
    metrics = []

    for (model, event, phase, th), group in df.groupby(["Model", "Event Type", "Phase", "Threshold"]):
        y_true = group["True"].values
        y_pred = group["Predicted"].values
        errors = group["Error"].values

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        coverage = group["Predicted"].sum() / len(group)
        
        tp_errors = errors[(y_true == 1) & (y_pred == 1)] # Calculate MAE across TPs only
        mae = np.nan if len(tp_errors) == 0 else np.nanmean(tp_errors)

        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        metrics.append({
            "Model": model,
            "Event Type": event,
            "Phase": phase,
            "Threshold": th,
            "Count": len(group),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Coverage": coverage,
            "MAE": mae,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        })

    return pd.DataFrame(metrics)


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
    
    

def compare_preds(trained_model, org_model, sample):
    trained_model.to_preferred_device()
    org_model.to_preferred_device()
        
    trained_model.eval()
    org_model.eval()
    
    fig = plt.figure(figsize=(15, 7))
    axs = fig.subplots(
        4, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [2,1,2,2]}
    )
    for ax in axs[1:]: ax.set_ylim(-0.05, 1.05)
    
    # Plot sample waveform
    axs[0].plot(sample["X"].T, label=["Z", "N", "E"])
    axs[0].text(0.05, 0.1, "sample waveform", transform=axs[0].transAxes, ha="left", va="top")
        
    
    # Plot ground truth probability labels
    axs[1].plot(sample["y"].T, label=list(trained_model.labels))
    axs[1].text(0.05, 0.2, "ground truth label (probability)", transform=axs[1].transAxes, ha="left", va="top")
    
    # Plot predictions from trained model
    with torch.no_grad():
        x = torch.tensor(sample["X"]).float().to(trained_model.device).unsqueeze(0)
        pred = trained_model(x) 
        pred = trained_model.annotate_batch_post(pred, None, trained_model.get_model_args())
        pred = pred.permute(0, 2, 1).squeeze().detach().numpy()

    axs[2].plot(pred.T, label=list(trained_model.labels))
    axs[2].text(0.05, 0.1, "trained model pred", transform=axs[2].transAxes, ha="left", va="top")
    
    # Plot predictions from original model
    with torch.no_grad():
        x = torch.tensor(sample["X"]).float().to(org_model.device).unsqueeze(0)
        pred = org_model(x) 
        pred = org_model.annotate_batch_post(pred, None, org_model.get_model_args())
        pred = pred.permute(0, 2, 1).squeeze().detach().numpy()
    
    axs[3].plot(pred.T, label=list(trained_model.labels))
    axs[3].text(0.05, 0.1, "original model pred", transform=axs[3].transAxes, ha="left", va="top")
    for ax in axs: ax.legend()
    
    