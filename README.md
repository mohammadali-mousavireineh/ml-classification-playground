# ML Classification Playground

A small place to test classification models before moving ideas into the main IoT paper project.

This repo is connected to my IoT/Windows classification work. The goal here is simple: try baseline models, compare metrics, and keep the experiments clean enough to repeat later.

## Why this repo exists

In the main research project, the dataset and pipeline can get heavy. This repository is for quick experiments:

- loading tabular network/security data
- trying a few baseline classifiers
- checking accuracy, precision, recall, F1-score, and confusion matrix
- comparing models before using them in the main paper pipeline
- keeping small notes about what worked and what did not

## Current models

- Logistic Regression
- Random Forest
- Gradient Boosting

More models can be added later, for example SVM, XGBoost, LightGBM, or a small neural network.

## Project structure

```text
ml-classification-playground/
├── configs/              # experiment settings
├── data/                 # small sample data only
├── docs/                 # notes and paper connection
├── notebooks/            # exploratory notebooks
├── reports/              # saved results and figures
├── scripts/              # simple command-line scripts
├── src/ml_playground/    # reusable Python code
└── tests/                # small checks
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the baseline experiment

```bash
python scripts/run_baseline.py --data data/sample_iot_windows.csv --target label
```

The output will be saved in `reports/`.

## Notes

The sample CSV is only a small artificial dataset. It is here so the code can run immediately. For real experiments, I will use the dataset from the main IoT/Windows classification project and keep the raw data out of GitHub if it is too large or not public.

## Next steps

- add the real feature list from the IoT paper project
- add cross-validation
- add LightGBM experiment
- save model comparison tables
- write a short result summary in `docs/experiment-log.md`

I use this repo to keep my early classification experiments organized before moving them into the main research pipeline.