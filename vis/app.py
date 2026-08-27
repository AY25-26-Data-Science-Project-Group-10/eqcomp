import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, math, os
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import app_utils as autils

st.set_page_config(layout="wide")

# Load samples from data folder
samples = autils.load_samples()

# ---------------------------------------------------------
# Sidebar — Filters
# ---------------------------------------------------------
st.sidebar.title("Filters")

event_type = st.sidebar.selectbox(
    "Event type",
    ["earthquakes", "explosions", "noise"]
)

phase_filter = None
if event_type != "noise":
    phase_filter = st.sidebar.selectbox("Phase", ["P", "S", "All"])

all_models = sorted({m for s in samples for m in s["predictions"].keys()})

model_filter = st.sidebar.multiselect(
    "Models",
    all_models,
    default=all_models
)

cf_filter = st.sidebar.radio(
    "Confusion-matrix filter",
    ["All", "TP", "FP", "FN", "TN"]
)

show_non_matching = st.sidebar.checkbox(
    "Show non-matching model outputs (greyed out)",
    value=True
)

# ---------------------------------------------------------
# Filter samples by event type
# ---------------------------------------------------------
valid_samples = [s for s in samples if 
                 autils.sample_matches_filters(s, event_type, phase_filter, model_filter, cf_filter)]


# ---------------------------------------------------------
# Page state: gallery or detailed view
# ---------------------------------------------------------
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# ---------------------------------------------------------
# Helper: Generate low-res waveform images preview tile 
# for faster loading of waveform gallery
# ---------------------------------------------------------
def preview_waveform(sample):
    X = sample["X"]
    fig = go.Figure()
    comps = ["Z", "N", "E"]

    for i in range(3):
        fig.add_trace(go.Scatter(
            y=X[i],
            mode="lines",
            line=dict(width=0.7),
            name=comps[i]
        ))

    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False
    )
    return fig

# ---------------------------------------------------------
# Helper: build stacked figure (waveform + model traces)
# ---------------------------------------------------------
def build_stacked_figure(sample, models_to_show, sample_height=0.4):
    X = sample["X"]
    true_picks = sample["true_picks"]

    n_rows = 1 + len(models_to_show)
    titles = ["Sample waveform"] + [f"Model: {m}" for m, _, _ in models_to_show]
    model_height = (1-sample_height) / len(models_to_show) if len(models_to_show) > 0 else 1

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        row_heights=[sample_height] + [model_height] * len(models_to_show),
        shared_xaxes=True,
        print_grid=True,
        vertical_spacing=0.02,
        subplot_titles=titles
    )

    # Row 1: Sample waveform
    comps = ["Z", "N", "E"]
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                y=X[i],
                mode="lines",
                name=f"Waveform {comps[i]}",
                line=dict(width=0.5)
            ),
            row=1,
            col=1
        )
    for ph, t in true_picks.items():
        if not math.isnan(t): # None becomes math.nan in plotly
            fig.add_vline( # Display true picks
                x=t,
                line_width=2,
                line_color="green",
                annotation_text=f"True {ph}",
                row=1,
                col=1
            )

    # Rows 2..N: model probability traces
    for idx, (model_name, pred, grey_out) in enumerate(models_to_show, start=2):
        probs = pred["probs"]
        picks = pred["picks"]
        errors = pred["errors"]

        def grey(color):
            return "lightgrey" if grey_out else color

        # Unique legend group per subplot
        legend_group = f"model_{idx}"
        
        if isinstance(probs, dict):
            for ph, trace in probs.items():
                fig.add_trace(
                    go.Scatter(
                        y=trace,
                        mode="lines",
                        name=f"{model_name} {ph}",
                        showlegend=True,
                        line=dict(color=grey(autils.PROB_COLOURS.get(ph.upper(), "blue")))
                    ),
                    row=idx,
                    col=1
                )   
        else:
            labels = ["N", "P", "S"]
            for i in range(probs.shape[0]):
                ph = labels[i]
                fig.add_trace(
                    go.Scatter(
                        y=probs[i],
                        mode="lines",
                        name=f"{model_name} {ph} prob",
                        legendgroup=legend_group,
                        showlegend=True,
                        line=dict(color=grey(autils.PROB_COLOURS[ph]))
                    ),
                    row=idx,
                    col=1
                )

        for ph, t in picks.items():
            if t is not None:
                error = errors[ph]
                fig.add_vline(
                    x=t,
                    line_width=2,
                    line_color=grey("red"),
                    annotation_text=f"{model_name} {ph} error:{error:3f}s",
                    annotation_yshift=-20 if ph =="S" else -40,
                    row=idx,
                    col=1
                )
        fig.update_yaxes(range=[-0.05, 1.05], row=idx, col=1)
    fig.update_layout(height=300 * n_rows, showlegend=True)
    
    return fig

# ---------------------------------------------------------
# PAGE 1 — WAVEFORM GALLERY
# ---------------------------------------------------------

if st.session_state.selected_sample is None:
    st.header(f"Showing {len(valid_samples)} results")

    cols = st.columns(6)

    for i, sample in enumerate(valid_samples):
        thumb = autils.generate_thumbnail(sample["X"])

        with cols[i % 6]:
            st.image(thumb, width='stretch')
            button_text = f"#{sample['index']} {sample["station_network_code"]}.{sample["station_code"]}.{sample["trace_channel"]}"
            if st.button(button_text, key=f"btn_{i}"):
                st.session_state.selected_sample = sample
                st.rerun()

# ---------------------------------------------------------
# PAGE 2 — DETAILED VIEW
# ---------------------------------------------------------
else:
    sample = st.session_state.selected_sample
    header_text = f'{sample["trace_start_time"]} {sample["station_network_code"]}.{sample["station_code"]}.{sample["trace_channel"]}'

    col1, col2 = st.columns([0.8, 0.2])

    # Header and back button
    with col1:
        st.header(header_text)

    with col2:
        st.write("")   # pushes button down to align visually
        if st.button("Back to search results", key="back_top"):
            st.session_state.selected_sample = None
            st.rerun()
    # Show sample waveform and model output probability traces
    models_to_show = autils.filter_models(sample, model_filter, phase_filter, cf_filter, show_non_matching)
    fig = build_stacked_figure(sample, models_to_show)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show metadata table
    metadata_dict = {field: sample.get(field, "N/A") for field in autils.metadata_fields}
    st.subheader("Sample metadata")
    st.table(metadata_dict)

    # Back button
    if st.button("Back to search results", key="back_bottom"):
        st.session_state.selected_sample = None
        st.rerun()