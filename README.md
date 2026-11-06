# Automated ML Trainer: Telco Customer Churn

An automated machine learning pipeline that predicts which telecom customers are likely to churn. It loads customer records from SQLite, preprocesses them, compares five candidate models with stratified cross-validation, selects a champion, evaluates it once on a held-out test set, and saves the champion as a single file.

This project is the middle part of a three-project series:

- **MemStream-DB** (data layer): memory-conscious record access from SQLite.
- **Automated ML Trainer** (this repository): training, selection, and serialization.
- **FastAPI Serving** (in progress): loads the saved champion and exposes a `/predict` endpoint.

## Dataset

- **Source:** IBM Telco Customer Churn dataset (downloaded once by `db_loader.py`).
- **Size:** 7,043 customers, 19 features after dropping `customerID`.
- **Target:** `Churn` (`Yes` / `No`). About 26% of customers churn.
- **Split:** 80/20 stratified train/test split, `random_state=42`. Train: 5,634 rows. Test: 1,409 rows.

## Results

### Cross-validation (training data only, 5-fold stratified, churn-class F1)

| Model | CV F1 (mean ± std) |
|---|---|
| Dummy baseline (stratified random) | 0.276 ± 0.025 |
| Logistic Regression | 0.630 ± 0.021 |
| Decision Tree | 0.619 ± 0.018 |
| Random Forest | 0.633 ± 0.014 |
| Gradient Boosting | 0.634 ± 0.022 |

The four real models fall within one standard deviation of each other. Their differences are within fold-to-fold noise, so no single model is clearly better. The dummy baseline confirms the models learn real signal.

### Champion: Logistic Regression (held-out test set, 1,409 customers)

| Metric | Value |
|---|---|
| Churn F1 | 0.62 |
| Churn recall | 0.79 (294 of 374 churners caught) |
| Churn precision | 0.51 |
| ROC-AUC | 0.84 |
| Accuracy | 0.74 |

**Why accuracy is not the headline metric:** always predicting "No" gives 73.5% accuracy on this test set. The champion's 74% is only slightly better than that, so churn F1 and recall are the metrics to watch.

## Pipeline

1. **Load:** `db_loader.py` downloads the CSV if missing, loads it into `data/warehouse.db`, and streams rows back out.
2. **Prepare:** `data_pipeline.py` drops `customerID`, separates the target, identifies numeric and categorical columns, and performs the stratified split.
3. **Train and select:** `model_engine.py` wraps each candidate in a scikit-learn Pipeline, so scaling and encoding are fitted inside each CV fold. Hyperparameters are tuned by grid search over the training data.
4. **Report:** each champion candidate is evaluated once on the test set for reporting. The test set is not used for selection.
5. **Save:** the selected champion pipeline (preprocessing plus model) is saved to `models/champion_model.pkl`.

## Project structure

```text
automated_ml_trainer/
├── data/                    # generated: telco_churn.csv, warehouse.db
├── models/                  # generated: champion_model.pkl
├── src/
│   ├── __init__.py
│   ├── db_loader.py         # download, load, stream records from SQLite
│   ├── data_pipeline.py     # load, split, column identification
│   ├── model_engine.py      # candidates, CV tuning, evaluation, selection
│   └── logger.py            # shared logging setup
├── main.py                  # runs the full training pipeline
├── evaluate_champion.py     # evaluates the saved champion on the full test set
├── requirements.txt
└── .gitignore
```

## Setup

Requires Python and a virtual environment.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run from the project root (the folder that contains `main.py`):

```powershell
python main.py              # download/load data, train, select, save champion
python evaluate_champion.py # evaluate the saved champion on the full test set
```

Logs are written to `app.log`. The `Model comparison` section shows the CV and test numbers for every candidate.

## Design decisions

- **No data leakage in preprocessing:** scalers and encoders are part of the Pipeline, so they are refitted inside each CV fold.
- **Selection uses CV only:** the champion is chosen by mean cross-validated F1 on the churn class. Test metrics are reported but never used to choose.
- **Refit on training data only:** the saved champion is trained on the 80% training split. Refitting on the full dataset would make the test numbers optimistic.
- **Tie-break rule:** if several models are within one CV standard deviation of the top score, the simplest one wins. Simplicity is defined by candidate order in `model_engine.py`, not by a measured complexity score.
- **Blank `TotalCharges` becomes 0.0:** these are tenure-0 customers, who have no billed total yet.
- **Failures are isolated:** if one candidate fails, it is logged as CRITICAL and skipped, and the others still run.

## Limitations

- Records stream from SQLite, but training collects them into memory first. This is fine for 7,000 rows; larger data would need chunked processing.
- One dataset and one train/test split. Results would benefit from repeated CV or nested CV.
- Classification uses the default 0.5 probability threshold. Threshold tuning is not implemented yet.
- Hyperparameter grids are small. Some best values sit at the edge of the grid, so wider searches may help a little.
- The "simplest model" ordering is hard-coded.
- No monitoring or data-drift detection yet.

## Roadmap

- Tune the decision threshold on out-of-fold probabilities, and save it alongside the model.
- Serve the champion through a FastAPI `/predict` endpoint.
- Load training data directly from the warehouse without the CSV download step.