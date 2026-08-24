# Performance of pretrained models

This report compares the performance of EQCCT(P/S), EQTransformer, and PhaseNet models as-is from Seisbench.

## Dataset
The number of 3-component waveforms used for each class:
* Earthquakes: 288 waveforms
* Explosions: 699 waveforms
* Noise: 1278 waveforms

The Finnish dataset has randomised window start times and downsamples high frequency waveforms (>100Hz) to 100 Hz. 

The validation (dev) split was used for this report.

## Earthquakes P-phase picking
| Model         | Event Type   | Phase   |   Threshold |   Count | Precision   | Recall    | F1        | Coverage   | MAE       |   TP |   TN |   FP |   FN |
|:--------------|:-------------|:--------|------------:|--------:|:------------|:----------|:----------|:-----------|:----------|-----:|-----:|-----:|-----:|
| EQCCTP        | earthquakes  | P       |         0.2 |     288 | **1.000**   | 0.010     | 0.021     | 0.010      | **0.047** |    3 |    0 |    0 |  285 |
| EQCCTS        | earthquakes  | P       |         0.2 |     288 | 0.000       | 0.000     | 0.000     | 0.000      | nan       |    0 |    0 |    0 |  288 |
| EQTransformer | earthquakes  | P       |         0.2 |     288 | **1.000**   | **0.278** | **0.435** | **0.278**  | 0.059     |   80 |    0 |    0 |  208 |
| PhaseNet      | earthquakes  | P       |         0.2 |     288 | **1.000**   | 0.215     | 0.354     | 0.215      | 0.080     |   62 |    0 |    0 |  226 |


## Earthquakes S-phase picking
| Model         | Event Type   | Phase   |   Threshold |   Count | Precision   | Recall    | F1        | Coverage   | MAE       |   TP |   TN |   FP |   FN |
|:--------------|:-------------|:--------|------------:|--------:|:------------|:----------|:----------|:-----------|:----------|-----:|-----:|-----:|-----:|
| EQCCTP        | earthquakes  | S       |         0.2 |     288 | 0.000       | 0.000     | 0.000     | 0.000      | nan       |    0 |    0 |    0 |  288 |
| EQCCTS        | earthquakes  | S       |         0.2 |     288 | **1.000**   | 0.028     | 0.054     | 0.028      | **0.042** |    8 |    0 |    0 |  280 |
| EQTransformer | earthquakes  | S       |         0.2 |     288 | **1.000**   | **0.191** | **0.321** | **0.191**  | 0.074     |   55 |    0 |    0 |  233 |
| PhaseNet      | earthquakes  | S       |         0.2 |     288 | **1.000**   | 0.094     | 0.171     | 0.094      | 0.082     |   27 |    0 |    0 |  261 |


## Explosions P-phase picking
| Model         | Event Type   | Phase   |   Threshold |   Count | Precision   | Recall    | F1        | Coverage   | MAE       |   TP |   TN |   FP |   FN |
|:--------------|:-------------|:--------|------------:|--------:|:------------|:----------|:----------|:-----------|:----------|-----:|-----:|-----:|-----:|
| EQCCTP        | explosions   | P       |         0.2 |     699 | **1.000**   | 0.003     | 0.006     | 0.003      | 0.127     |    2 |    0 |    0 |  697 |
| EQCCTS        | explosions   | P       |         0.2 |     699 | 0.000       | 0.000     | 0.000     | 0.000      | nan       |    0 |    0 |    0 |  699 |
| EQTransformer | explosions   | P       |         0.2 |     699 | **1.000**   | 0.182     | 0.308     | 0.182      | 0.090     |  127 |    0 |    0 |  572 |
| PhaseNet      | explosions   | P       |         0.2 |     699 | **1.000**   | **0.185** | **0.312** | **0.185**  | **0.078** |  129 |    0 |    0 |  570 |


## Explosions S-phase picking
| Model         | Event Type   | Phase   |   Threshold |   Count | Precision   | Recall    | F1        | Coverage   | MAE       |   TP |   TN |   FP |   FN |
|:--------------|:-------------|:--------|------------:|--------:|:------------|:----------|:----------|:-----------|:----------|-----:|-----:|-----:|-----:|
| EQCCTP        | explosions   | S       |         0.2 |     699 | 0.000       | 0.000     | 0.000     | 0.000      | nan       |    0 |    0 |    0 |  699 |
| EQCCTS        | explosions   | S       |         0.2 |     699 | **1.000**   | 0.021     | 0.042     | 0.021      | **0.084** |   15 |    0 |    0 |  684 |
| EQTransformer | explosions   | S       |         0.2 |     699 | **1.000**   | **0.096** | **0.175** | **0.096**  | 0.097     |   67 |    0 |    0 |  632 |
| PhaseNet      | explosions   | S       |         0.2 |     699 | **1.000**   | 0.072     | 0.134     | 0.072      | 0.107     |   50 |    0 |    0 |  649 |


## Noise P-phase picking
| Model         | Event Type   | Phase   |   Threshold |   Count |   Precision |   Recall |   F1 | Coverage   |   MAE |   TP |   TN |   FP |   FN |
|:--------------|:-------------|:--------|------------:|--------:|------------:|---------:|-----:|:-----------|------:|-----:|-----:|-----:|-----:|
| EQCCTP        | noise        | P       |         0.2 |    1278 |           0 |        0 |    0 | 0.387      |   nan |    0 |  783 |  495 |    0 |
| EQCCTS        | noise        | P       |         0.2 |    1278 |           0 |        0 |    0 | **0.0**    |   nan |    0 | 1278 |    0 |    0 |
| EQTransformer | noise        | P       |         0.2 |    1278 |           0 |        0 |    0 | 0.002      |   nan |    0 | 1276 |    2 |    0 |
| PhaseNet      | noise        | P       |         0.2 |    1278 |           0 |        0 |    0 | 0.027      |   nan |    0 | 1243 |   35 |    0 |


## Noise S-phase picking
| Model         | Event Type   | Phase   |   Threshold |   Count |   Precision |   Recall |   F1 | Coverage   |   MAE |   TP |   TN |   FP |   FN |
|:--------------|:-------------|:--------|------------:|--------:|------------:|---------:|-----:|:-----------|------:|-----:|-----:|-----:|-----:|
| EQCCTP        | noise        | S       |         0.2 |    1278 |           0 |        0 |    0 | **0.0**    |   nan |    0 | 1278 |    0 |    0 |
| EQCCTS        | noise        | S       |         0.2 |    1278 |           0 |        0 |    0 | 0.003      |   nan |    0 | 1274 |    4 |    0 |
| EQTransformer | noise        | S       |         0.2 |    1278 |           0 |        0 |    0 | 0.002      |   nan |    0 | 1276 |    2 |    0 |
| PhaseNet      | noise        | S       |         0.2 |    1278 |           0 |        0 |    0 | 0.025      |   nan |    0 | 1246 |   32 |    0 |

## Conclusion
At a threshold of 0.2s, **EQTransformer** outperforms the other models in earthquake P and S, and explosion S-picking. For other event-type-phase splits, it was was the second-best model whose performance was almost identical to the best model. Therefore, **EQTransformer** is the ideal candidate for finetuning.

## Code for reproduction
```python
import seisbench.data as sbd

import os
from pathlib import Path
import train_utils as tutils
import model_utils as mutils

DIR_DATA = Path(os.getcwd()) / "data"

# Load the train validation and test splits
dataset = sbd.WaveformDataset(DIR_DATA)
train_dataset, dev_dataset, test_dataset = dataset.train_dev_test()

# Load pre-trained model from seisbench Model weights
configs = {"EQTransformer": mutils.EQTransformerConfig(),
           "EQCCTP": mutils.EQCCTConfig("P"),
           "EQCCTS": mutils.EQCCTConfig("S"),
           "PhaseNet": mutils.PhaseNetConfig()
        }

# Compare performance of the pretrained baseline models
results = []
for model_name, config in configs.items():
    train_gen, dev_gen, test_gen = tutils.preprocess_data(config, train_dataset, dev_dataset, test_dataset)
    model = config.get_new_model()
    model.to_preferred_device()

    results += tutils.evaluate_model(model, dev_gen, dev_dataset, model_name, mae_thresholds=[0.2])

metrics = tutils.aggregate_metrics(results)
tutils.generate_metric_report(metrics)
```