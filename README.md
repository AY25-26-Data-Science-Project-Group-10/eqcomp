# EQcomp
Unified Evaluation and Fine‑Tuning Framework for Seismic Phase Pickers on the Finnish Dataset


## Motivation
The original model repositories—EQTransformer and EQCCT—were forked from upstream sources and evolved independently. As a result:

* They used different dataset formats (MiniSEED vs HDF5 vs custom structures).
* They required different file system layouts and preprocessing conventions.
* They depended on different Python environments, often with conflicting dependencies.
* They lacked a unified interface for:
    * loading pretrained weights
    * fine‑tuning
    * comparing models on the same dataset
    * executing a reproducible evaluation pipeline

This fragmentation forced users to spend time resolving environment issues instead of conducting scientific experiments. EQcomp addresses all of this issues.


## What EQcomp offers

### 1. Reduced setup time
EQComp accelerates environment configuration by adopting SeisBench’s standardised APIs for every model. Numerous of phase pickers, such as PhaseNet, EQTransformer, EQCCT, can now be evaluated within the same PyTorch environment using the same dataset and metrics calculation. 

### 2. One environment, multiple models 
EQcomp builds on Seisbench's unified interface, which is based solely on Pytorch. Therefore, there is no need to struggle with using both Pytorch and Tensorflow environments.

### 3. Fine-tuning support for all models
Some upstream repositories (e.g., EQCCT) do not expose their training pipelines. EQComp reconstructs these pipelines by using SeisBench’s interface, which provides direct, low-level access to the Pytorch implementation of the models.


## Repository Structure

```
eqcomp/
├── benchmarks/    # Model comparison reports
├── data/          # Finnish dataset: metadata.csv, waveforms.hdf5 (too big to upload for now)
├── docs/          # Documentation
├── experiments/   # Model artifacts, metadata, metrics from finetuning experiments
├── utils/
│   ├── config.py  # File paths, training, and dataset settings
│   ├── eval.py    # Helper functions for model comparison 
│   ├── model.py   # Models bundled with their training configurations
│   └── train.py   # Helper functions for fine-tuning and evaluation
├── vis.           # Waveform and model output visualisation dashboard
└── finetune.ipynb # Example notebook for finetuning          
```

## Set up the Python environment (notebooks + scripts)

Run these commands from the repository root (works on Linux/macOS).

### 1. Create a virtual environment
```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


## Adding a New Model
This section details instructions on how to integrate a new Seisbench phase-picking model into EQcomp.

### 1. Subclass ModelConfig
In `utils/model.py`, create a new class:

```python
class MyModelConfig(ModelConfig):
    ...
```
You must implement all methods decorated with `@abstractmethod`, including:
* `get_new_model()`
* `get_model_class()`
* `loss_fn()` (Note: please infer this from the model's paper and/or original codebase)

### 2. Define the Preprocessing Steps
The Finnish training dataset consists of raw waveforms because each model has its own data preprocessing steps. This information is not available on Seisbench, so you have to infer from the model's original repository.

Inside `__init__()`, define the model-specific waveform preprocessing steps, such as normalisation, demeaning, detrending, filtering. The steps should be defined in a `list`, and assigned to `self._augs`. 

```python
class PhaseNetConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self._augs = [ 
            sbg.Normalize(
                demean_axis=-1, 
                amp_norm_axis=-1, amp_norm_type='std', 
                eps=EPS, key='X'),
            sbg.ProbabilisticLabeller(
                shape="gaussian", 
                label_columns=ps_phase_dict, model_labels=["N", "P", "S"], 
                dim=0, sigma=20),
        ]
```

### 3. Validate Input Sample Requirements
Different models expect different input lengths.
For example:
* EQCCT → fixed 6000 samples

* EQTransformer → fixed 6000 samples

* PhaseNet → typically 3000–6000 samples

* DKPN → fixed in_samples (often 400)


The Finnish dataset uses 6000 samples per waveform, so you must ensure: the model’s expected input length matches 6000 or implement a resampling/windowing strategy. EQComp does not enforce this automatically, so the user must exercise discretion.

## TODO
1. ~~Implement/copy paste loss functions~~
2. Visualisation of dataset and prediction results
3. ~~Compare prediction results of fine-tuned vs non fine-tuned models~~
4. ~~Downsample high-frequency traces to 100hz to increase dataset size~~
5. Configure GPU training
6. Hyperparameter tuning