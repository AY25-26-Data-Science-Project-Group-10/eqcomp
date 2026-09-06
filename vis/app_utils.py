import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, math, os
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import plotly.express as px

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
DIR_VIS_DATA = 'data'
FILENAME_PKL = "samples.pkl"


# ---------------------------------------------------------
# Data pickle settings
# ---------------------------------------------------------
FILENAME_PREFIX = "samples_part"
CHUNK_SIZE = 300   # Calibrated to be just under 100MB for github


# ---------------------------------------------------------
# Colour settings
# --------------------------------------------------------
PROB_COLOURS = {
    "N": "cornflowerblue",   # blue
    "DETECTION": "cornflowerblue",   # blue
    "P": "orange",   # orange
    "S": "lightgreen",    # green
    "UNSELECTED": "lightgrey"
}

PICK_COLOURS = {
    "P": "red",
    "S": "green",
    "UNSELECTED": "lightgrey"
}

WAVE_COLOURS = px.colors.qualitative.Dark2

# ---------------------------------------------------------
# Dashboard data
# --------------------------------------------------------
metadata_fields = [
    "event_type",
    "index", # Used to search in metadata.csv
    "source_id",
    "source_origin_time",
    "source_depth_km",
    "source_magnitude",
    "station_network_code",
    "station_code",
    "trace_channel",
    "trace_start_time"
]



# ---------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------
@st.cache_resource
def load_samples():
    """Load unified sample objects"""
    samples = []
    for pkl_file in sorted(Path(DIR_VIS_DATA).glob(f"{FILENAME_PREFIX}_*.pkl")):
        with open(pkl_file, "rb") as f:
            part = pickle.load(f)
            samples.extend(part)
    return samples

def sample_matches_filters(sample, event_type, phase_filter, model_filter, cf_filter):
    """
    Select waveforms to preview in gallery

    Returns True if the sample matches the current filter settings.
    Used to determine which samples appear in the waveform gallery.
    """
    # 1. Event type filter
    if sample["event_type"] != event_type:
        return False
    
        # 2. Phase filter (skip if "No filter")
    if phase_filter != "No filter":
        phase_ok = False

        for model_name in model_filter:
            pred = sample["predictions"][model_name]
            picks = pred["picks"]
            tp_fp_fn = pred["tp_fp_fn"]
            
            # Model cannot output this phase → skip
            if phase_filter not in picks.keys():
                continue

            # Model CAN output this phase
            phase_cf = tp_fp_fn.get(phase_filter)

            # Confusion matrix filter
            if cf_filter == "No filter":
                phase_ok = True
                break
            elif phase_cf == cf_filter:
                phase_ok = True
                break

        if not phase_ok:
            return False

    return True

@st.cache_resource
def generate_thumbnail(X, true_picks=None):
    """Generate a small PNG thumbnail for a waveform."""
    fig, ax = plt.subplots(figsize=(3, 1))  # small thumbnail

    # Plot Z component
    ax.plot(X[0], linewidth=0.5, color="black")

    # Draw true picks if provided
    if true_picks is not None:
        P = true_picks.get("P", None)
        S = true_picks.get("S", None)

        if P is not None and np.isfinite(P):
            ax.axvline(P, color=PICK_COLOURS["P"], linewidth=1.2)  # orange for P

        if S is not None and np.isfinite(S):
            ax.axvline(S, color=PICK_COLOURS["S"], linewidth=1.2)  # green for S

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    plt.tight_layout()
    plt.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return buf


def filter_models(sample, model_filter, phase_filter, cf_filter, show_all_models, all_models):
    """Determines whether a model probability trace output should be displayed"""
    models_to_show = []

    for model_name in all_models:
        pred = sample["predictions"][model_name]
        picks = pred["picks"]
        tp_fp_fn = pred["tp_fp_fn"]

        show_model = True   # default
        grey_out = False    # default
        
        
        if phase_filter != "No filter":
            if phase_filter not in picks:
                show_model = False
            else:
                if cf_filter != "No filter":
                    phase_cf = tp_fp_fn.get(phase_filter)
                    if phase_cf != cf_filter:
                        grey_out = True
                        show_model = False
    
        if model_name not in model_filter:
            show_model = False
            
        if show_all_models:
            show_model = True
            
        if show_model:
            models_to_show.append((model_name, pred, grey_out))

    return models_to_show