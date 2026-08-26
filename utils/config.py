from pathlib import Path
import os


# ----------- eqdet directories ----------- 
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

# ----------- eqcomp directories ----------- 
DIR_DATA = Path(os.getcwd()) / "data"
FILENAME_METADATA = "metadata.csv"
FILENAME_WAVEFORMS = "waveforms.hdf5"
FILENAME_STATIONS = "stations.xml"

FILENAME_MODEL = "model.pth"

SAMPLING_RATE = 100
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.10
WIN_LEN = 6000