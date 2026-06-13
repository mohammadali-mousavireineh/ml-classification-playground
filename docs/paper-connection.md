# Connection to the IoT/Windows Paper Project

This repository is not the main paper repository. It is a smaller space for testing classification ideas before moving them into the main IoT/Windows pipeline.

## Research direction

The main topic is tabular classification for IoT/Windows security data. In this playground, I focus on the basic steps that are easy to break in a research pipeline:

- clean train/test split
- no preprocessing leakage
- same metrics for all models
- simple baseline models before complex models
- saving result tables for comparison

## Why baseline models matter

Before using advanced models, I want to know how simple models behave. If a simple model gives a suspiciously high score, it may mean there is leakage or the split is not realistic.

This is useful for the paper because it helps answer questions like:

- Are the features too directly connected to the label?
- Is the model learning real patterns or dataset artifacts?
- Which metric is more useful: accuracy, recall, or F1-score?
- Which baseline should be reported before adding more advanced models?

## Planned experiment notes

| Experiment | Model | Dataset | Main question | Status |
|---|---|---|---|---|
| E01 | Logistic Regression | sample data | Does the pipeline run end-to-end? | done |
| E02 | Random Forest | paper dataset | Is the result stable after split? | planned |
| E03 | LightGBM | paper dataset | Does boosting improve F1-score? | planned |
| E04 | Leakage check | paper dataset | Are there suspicious columns? | planned |

## Feature ideas

The sample dataset uses placeholder features such as:

- duration
- packet count
- total bytes
- mean packet size
- packet rate
- failed connections
- source port
- protocol

For the paper dataset, I should replace these with the real columns and write the final feature list here.
