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


def preprocess_data(config, train_dataset, dev_dataset, test_dataset):
    train_generator = sbg.GenericGenerator(train_dataset)
    dev_generator = sbg.GenericGenerator(dev_dataset)
    test_generator = sbg.GenericGenerator(test_dataset)

    train_generator.add_augmentations(config.get_augs())
    dev_generator.add_augmentations(config.get_augs())
    test_generator.add_augmentations(config.get_augs())
    
    return train_generator, dev_generator, test_generator

def get_true_pick(sample):
    return int(np.argmax(sample["y"]))  # true pick time from probability trace

def get_predicted_pick(prob_trace, threshold=0.1):
    peak_idx = int(np.argmax(prob_trace))
    peak_val = prob_trace[peak_idx]
    return peak_idx if peak_val > threshold else None

def evaluate_model(model, test_generator, test_dataset, model_name, mae_thresholds=[0.2, 0.5, 1.0]):
    results = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(test_generator))):
            sample = test_generator[i]
            meta = test_dataset.get_sample(i)[1]

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
                            error = (pred_t - true_t) / SAMPLING_RATE
                            y_pred = 1 if abs(error) <= th else 0

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
        
        # Signed bias (mean signed error)
        bias = np.nan if len(tp_errors) == 0 else np.nanmean(tp_errors)
        # MAE
        mae = np.nan if len(tp_errors) == 0 else np.nanmean(np.abs(tp_errors))

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
            "Bias": bias,
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
    
    
def generate_markdown(df_metrics):
    metric_cols_max = ["Precision", "Recall", "F1", "Coverage"]
    metric_cols_min = ["MAE"]

    df_md = df_metrics.copy()

    # 1. Round all metric columns to 3dp using pandas
    all_metric_cols = metric_cols_max + metric_cols_min
    df_md[all_metric_cols] = df_md[all_metric_cols].round(3)

    # 2. Compute rounded maxima and minima
    rounded_max = {col: df_md[col].max() for col in metric_cols_max}
    rounded_min = {col: df_md[col].min() for col in metric_cols_min}
    
    # 3. Convert all metric columns to formatted strings (3dp)
    # Special rule for noise: ONLY Coverage is bolded, and only the minimum
    if (df_metrics["Event Type"] == "noise").any():
        cov_min = df_md["Coverage"].astype(float).min()
        df_md["Coverage"] = df_md["Coverage"].apply(
            lambda x: f"**{x}**" if float(x) == cov_min else x
        )
        return df_md.to_markdown(index=False)
    
    for col in metric_cols_max + metric_cols_min:
        df_md[col] = df_md[col].apply(lambda x: f"{x:.3f}")

    # 4. Bold maxima
    for col in metric_cols_max:
        max_val = rounded_max[col]
        if max_val > 0:
            df_md[col] = df_md[col].apply(
                lambda x: f"**{x}**" if float(x) == max_val else x
            )

    # 5. Bold minima
    for col in metric_cols_min:
        min_val = rounded_min[col]
        df_md[col] = df_md[col].apply(
            lambda x: f"**{x}**" if float(x) == min_val else x
        )

    return df_md.to_markdown(index=False)


def generate_metric_report(df_metrics):
    event_phase_pairs = [
    ("earthquakes", "P"),
    ("earthquakes", "S"),
    ("explosions", "P"),
    ("explosions", "S"),
    ("noise", "P"),
    ("noise", "S"),
    ]

    for event, phase in event_phase_pairs:
        subset = df_metrics[(df_metrics["Event Type"] == event) &
                        (df_metrics["Phase"] == phase)]
        print(f"## {event.title()} {phase}-phase picking")
        print(generate_markdown(subset), end="\n\n\n")


def plot_pick_error_dist(results: dict, xlim=None, bins=20, fontsize=20):
    
    df = pd.DataFrame(results)

    # Only keep rows with numeric errors
    df_nonan = df[df["Error"].notna()].copy()
    
    # Model lists
    all_models = sorted(df_nonan["Model"].unique())
    models_P = [m for m in all_models if m != "EQCCTS"]   # EQCCTS cannot produce P
    models_S = [m for m in all_models if m != "EQCCTP"]   # EQCCTP cannot produce S

    event_types = ["earthquakes", "explosions"]
    phases = ["P", "S"]
    
    ncols = max(len(models_P), len(models_S))

    fig, axes = plt.subplots(
        nrows=len(event_types) * len(phases),   
        ncols=ncols,         
        figsize=(8 * ncols, 20),
        sharex=True,
        sharey=True
    )
     # Apply x-limits only if provided
    if xlim is not None:
        if not (isinstance(xlim, tuple) and len(xlim) == 2):
            raise ValueError("xlim must be a tuple of (xmin, xmax) or None.")
        plt.xlim(xlim)
        
    row_idx = 0
    for etype in event_types:
        for phase in phases:

            # Select correct model list
            models = models_P if phase == "P" else models_S
            
            for col_idx, model in enumerate(models):

                ax = axes[row_idx, col_idx]
                
                sub = df_nonan[
                    (df_nonan["Event Type"] == etype) &
                    (df_nonan["Phase"] == phase) &
                    (df_nonan["Model"] == model)
                ]

                if len(sub) == 0:
                    ax.set_title(f"{etype.capitalize()} — {phase} — {model} (no picks)")
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    continue
                
                tp_mask = (sub["True"] == 1) & (sub["Predicted"] == 1) # Calculate MAE over TPs only
                errors = sub["Error"].values[tp_mask]

                # Metrics
                mae = np.mean(np.abs(errors))
                sigma = np.std(errors)
                bias = np.mean(errors)

                # Histogram centered around zero
                ax.hist(
                    errors, 
                    bins=bins,
                    edgecolor="black",
                )

                ax.set_title(f"{etype.capitalize()} - {phase} - {model}", fontsize=fontsize)
                ax.set_xlabel("Picking Error (s)", fontsize=fontsize)
                ax.set_ylabel("Count", fontsize=fontsize)
                ax.tick_params(which='both', labelbottom=True, labelleft=True)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
                
                # Annotate metrics
                ax.text(0.05, 0.90, f"MAE = {mae:.3f}s", transform=ax.transAxes, fontsize=fontsize)
                ax.text(0.05, 0.80, f"σ = {sigma:.2f}s", transform=ax.transAxes, fontsize=fontsize)
                ax.text(0.05, 0.70, f"Bias = {bias:.3f}s", transform=ax.transAxes, fontsize=fontsize)
                ax.text(0.05, 0.60, f"Count = {len(errors)}", transform=ax.transAxes, fontsize=fontsize)

            row_idx += 1

    plt.tight_layout()
    plt.show()