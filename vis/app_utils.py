import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, math, os
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import app_utils as autils

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
DIR_VIZ_DATA = 'data'
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
    "N": "#1f77b4",   # blue
    "DETECTION": "#1f77b4",   # blue
    "P": "#ff7f0e",   # orange
    "S": "#2ca02c"    # green
}

PICK_COLOURS = {
    "P": "red",
    "S": "green" 
}

# ---------------------------------------------------------
# Data to display
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
    for pkl_file in sorted(Path(autils.DIR_VIZ_DATA).glob(f"{autils.FILENAME_PREFIX}_*.pkl")):
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

    # 2. Phase filter (skip if "All")
    if phase_filter not in (None, "All"):
        phase_ok = False

        for model_name in model_filter:
            pred = sample["predictions"][model_name]
            picks = pred["picks"]
            tp_fp_fn = pred["tp_fp_fn"]

            # Model cannot output this phase → skip
            if phase_filter not in picks:
                continue

            # Model CAN output this phase
            phase_cf = tp_fp_fn.get(phase_filter)

            # Confusion matrix filter
            if cf_filter == "All":
                phase_ok = True
                break
            else:
                if phase_cf == cf_filter:
                    phase_ok = True
                    break

        if not phase_ok:
            return False

    return True

@st.cache_resource
def generate_thumbnail(X):
    """Generate a small PNG thumbnail for a waveform."""
    fig, ax = plt.subplots(figsize=(3, 1))  # small thumbnail
    ax.plot(X[0], linewidth=0.5)            # only Z component for speed
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


def filter_models(sample, model_filter, phase_filter, cf_filter, show_non_matching):
    """Determines whether a model probability trace output should be displayed"""
    models_to_show = []

    for model_name in model_filter:
        pred = sample["predictions"][model_name]
        picks = pred["picks"]
        tp_fp_fn = pred["tp_fp_fn"]

        show_model = True
        grey_out = False

        if phase_filter not in (None, "All"):
            if phase_filter not in picks:
                show_model = False
            else:
                phase_cf = tp_fp_fn.get(phase_filter)

                if cf_filter == "All":
                    show_model = True
                else:
                    if phase_cf == cf_filter:
                        show_model = True
                    else:
                        if show_non_matching:
                            grey_out = True
                        else:
                            show_model = False

        if show_model:
            models_to_show.append((model_name, pred, grey_out))

    return models_to_show