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
# Initial setup
# ---------------------------------------------------------
st.set_page_config(layout="wide")

# Load samples from data folder
samples = autils.load_samples()

all_models = sorted({m for s in samples for m in s["predictions"].keys()})

# ---------------------------------------------------------
# Initialise session states
# ---------------------------------------------------------
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None


# ---------------------------------------------------------
# Sidebar — Filters
# ---------------------------------------------------------
st.sidebar.title("Filters")

def callback():
    st.session_state.selected_sample = None

event_type = st.sidebar.selectbox(
    key="event_type",
    label="Select Event type",
    options=["earthquakes", "explosions", "noise"],
    on_change=lambda: st.session_state.update(selected_sample=None),
)


phase_filter = st.sidebar.selectbox(
    key="phase_filter",
    label="Select phase",
    options=["P", "S", "No filter"],
    on_change=lambda: st.session_state.update(selected_sample=None),
)


cf_filter = None
if phase_filter != "No filter":
    options = None
    if event_type == "noise":
        options = ["No filter", "FP", "TN"]
    else:
        options = ["No filter", "TP", "FP", "FN", "TN"]
        
    cf_filter = st.sidebar.radio(
        key="cf_filter",
        label=f"Confusion-matrix filtering for phase {phase_filter}:",
        options=options,
        on_change=lambda: st.session_state.update(selected_sample=None),
    )

model_filter = st.sidebar.multiselect(
    key="model_filter",
    label="Select model(s)",
    options=all_models,
    on_change=lambda: st.session_state.update(selected_sample=None),
    default=all_models[0]
)

show_all_models = st.sidebar.checkbox(
    key="show_all_models",
    label="Show outputs of all models",
    value=True,
)

# ---------------------------------------------------------
# Filter samples by event type
# ---------------------------------------------------------
valid_samples = [s for s in samples if 
                 autils.sample_matches_filters(s, 
                                               st.session_state.event_type, 
                                               st.session_state.phase_filter, 
                                               st.session_state.model_filter, 
                                               st.session_state.cf_filter)]


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
                line=dict(width=0.5, color=autils.WAVE_COLOURS[i])
            ),
            row=1,
            col=1
        )
        
    # Print true picks on top of sample waveforms
    for ph, t in true_picks.items():
        if t is not None and np.isfinite(t): # Handle nan
            fig.add_vline( # Display true picks
                x=t,
                line_width=2,
                line_color=autils.PICK_COLOURS[ph.upper()],
                annotation_text=f"True {ph}",
                row=1,
                col=1
            )

    # Rows 2..N: model probability traces
    for idx, (model_name, pred, grey_out) in enumerate(models_to_show, start=2):
        probs = pred["probs"]
        picks = pred["picks"]
        errors = pred["errors"]

        # Unique legend group per subplot
        legend_group = f"model_{idx}"
        
        if isinstance(probs, dict):
            for ph, trace in probs.items():
                line_colour = autils.PROB_COLOURS["UNSELECTED"] if grey_out else autils.PROB_COLOURS[ph.upper()]
                fig.add_trace(
                    go.Scatter(
                        y=trace,
                        mode="lines",
                        name=f"{model_name} {ph}",
                        showlegend=True,
                        line=dict(color=line_colour)
                    ),
                    row=idx,
                    col=1
                )   
        else:
            labels = ["N", "P", "S"]
            for i in range(probs.shape[0]):
                ph = labels[i]
                line_colour = autils.PROB_COLOURS["UNSELECTED"] if grey_out else autils.PROB_COLOURS[ph.upper()]
                fig.add_trace(
                    go.Scatter(
                        y=probs[i],
                        mode="lines",
                        name=f"{model_name} {ph} prob",
                        legendgroup=legend_group,
                        showlegend=True,
                        line=dict(color=line_colour)
                    ),
                    row=idx,
                    col=1
                )
        # Print model picks on top of probability traces
        for ph, t in picks.items():
            if t is not None and np.isfinite(t): # Handle nan
                error = errors[ph]
                
                if error is not None and np.isfinite(t): # Handle nan
                    annot_text = f"{model_name} {ph}<br>error: {error:3f}s"
                else:
                    annot_text = f"{model_name} {ph}"
                line_colour = autils.PICK_COLOURS["UNSELECTED"] if grey_out else autils.PICK_COLOURS[ph.upper()]
                fig.add_vline(
                    x=t,
                    line_width=2,
                    line_color=line_colour,
                    annotation_text=annot_text,
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
        thumb = autils.generate_thumbnail(sample["X"], sample["true_picks"])

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
        if st.button("Back to search results", key="back_top", type="primary"):
            st.session_state.selected_sample = None
            st.rerun()
    # Show sample waveform and model output probability traces
    models_to_show = autils.filter_models(sample, model_filter, phase_filter, cf_filter, show_all_models, all_models)
    fig = build_stacked_figure(sample, models_to_show)
    st.plotly_chart(fig, width='stretch')

    # Show metadata table
    metadata_dict = {field: sample.get(field, "N/A") for field in autils.metadata_fields}
    st.subheader("Sample metadata")
    st.table(metadata_dict)

    # Back button
    if st.button("Back to search results", key="back_bottom", type="primary"):
        st.session_state.selected_sample = None
        st.rerun()