import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_pipeline import DataPipeline


MODEL_PATH = "models/champion_model.pkl"
POS_LABEL = "Yes"
NEG_LABEL = "No"


def load_held_out_split():
    # Same stratified split with random_state=42 as training.
    # X_test was never used to fit or select anything.
    pipeline = DataPipeline()
    pipeline.load_data_from_warehouse()
    pipeline.separate_features_target()
    pipeline.split_and_process()
    return pipeline.X_test, pipeline.y_test


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    classes = list(model.classes_)
    pos_index = classes.index(POS_LABEL)
    y_score = model.predict_proba(X_test)[:, pos_index]
    y_true_binary = (y_test == POS_LABEL).astype(int)

    print(f"Test rows: {len(y_test)} | actual churners: {int(y_true_binary.sum())}")

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"              pred {NEG_LABEL}  pred {POS_LABEL}")
    print(confusion_matrix(y_test, y_pred, labels=[NEG_LABEL, POS_LABEL]))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Summary (churn = positive class):")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_true_binary, y_score):.4f}")


def sanity_check_single_customer(model):
    # Not an evaluation. Only checks that the saved pipeline accepts a raw row
    # with the same columns the training data had.
    customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 1146.00,
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][list(model.classes_).index(POS_LABEL)]
    print(f"\nSanity check (one hand-made customer): predicted={prediction}, churn probability={probability:.3f}")


def main():
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {type(model).__name__}")
    print("Steps:", list(model.named_steps.keys()))

    X_test, y_test = load_held_out_split()
    evaluate(model, X_test, y_test)
    sanity_check_single_customer(model)


if __name__ == "__main__":
    main()