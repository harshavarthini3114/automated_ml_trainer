import os

import joblib
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.logger import get_production_logger


POS_LABEL = "Yes"
NEG_LABEL = "No"
BASELINE = "Dummy Baseline"
MODEL_PATH = "models/champion_model.pkl"


class ModelEngine:

    def __init__(self, n_splits=5, random_state=42):
        self.logger = get_production_logger()
        self.random_state = random_state

        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        self.scoring = {
            "f1": make_scorer(f1_score, pos_label=POS_LABEL),
            "recall": make_scorer(recall_score, pos_label=POS_LABEL),
            "precision": make_scorer(precision_score, pos_label=POS_LABEL, zero_division=0),
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
        }

        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.results = {}   # name -> {"cv": ..., "test": ..., "estimator": ...}
        self.order = []     # simple -> complex, used for tie-breaking
        self.best_model_name = None
        self.best_model = None

    def _candidates(self):
        rs = self.random_state
        return [
            (BASELINE, DummyClassifier(strategy="stratified", random_state=rs), {}),
            ("Logistic Regression", LogisticRegression(max_iter=1000), {
                "classifier__C": [0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
            }),
            ("Decision Tree", DecisionTreeClassifier(random_state=rs), {
                "classifier__max_depth": [3, 5, 10],
                "classifier__min_samples_leaf": [1, 5, 20],
                "classifier__class_weight": [None, "balanced"],
            }),
            ("Random Forest", RandomForestClassifier(random_state=rs), {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, None],
                "classifier__class_weight": [None, "balanced"],
            }),
            ("Gradient Boosting", HistGradientBoostingClassifier(random_state=rs), {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, None],
                "classifier__class_weight": [None, "balanced"],
            }),
        ]

    def run(self, preprocessor, X_train, X_test, y_train, y_test):
        self.preprocessor = preprocessor
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        self.logger.info(f"Train shape {X_train.shape}, test shape {X_test.shape}")

        for name, estimator, grid in self._candidates():
            self.order.append(name)
            try:
                self._run_one(name, estimator, grid)
            except Exception as error:
                self.logger.critical(f"{name} failed, skipping: {error}")

        if not self.results:
            raise RuntimeError("No model trained successfully.")

        self._log_comparison()
        self._select_champion()
        self._serialize_champion()

        return self.best_model_name

    def _run_one(self, name, estimator, grid):
        self.logger.info(f"{name}: tuning started")

        pipeline = Pipeline([
            ("preprocessor", clone(self.preprocessor)),
            ("classifier", clone(estimator)),
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=self.cv,
            scoring=self.scoring,
            refit="f1",
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(self.X_train, self.y_train)

        cv_summary = self._cv_summary(search)
        best = search.best_estimator_
        test_summary = self._evaluate_on_test(best)

        self.results[name] = {
            "cv": cv_summary,
            "test": test_summary,
            "estimator": best,
        }
        self._log_model(name)

    def _cv_summary(self, search):
        i = search.best_index_
        cv_results = search.cv_results_
        n_folds = self.cv.get_n_splits()

        summary = {
            "best_params": search.best_params_,
            "cv_f1_folds": [float(cv_results[f"split{k}_test_f1"][i]) for k in range(n_folds)],
            "cv_f1_mean": float(cv_results["mean_test_f1"][i]),
            "cv_f1_std": float(cv_results["std_test_f1"][i]),
        }
        for metric in ["recall", "precision", "accuracy", "roc_auc"]:
            summary[f"cv_{metric}_mean"] = float(cv_results[f"mean_test_{metric}"][i])

        return summary

    def _evaluate_on_test(self, model):
        classes = list(model.classes_)
        pos_index = classes.index(POS_LABEL)

        y_pred = model.predict(self.X_test)
        y_score = model.predict_proba(self.X_test)[:, pos_index]
        y_true_binary = (self.y_test == POS_LABEL).astype(int)

        cm = confusion_matrix(self.y_test, y_pred, labels=[NEG_LABEL, POS_LABEL])

        return {
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, pos_label=POS_LABEL, zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, pos_label=POS_LABEL, zero_division=0)),
            "f1": float(f1_score(self.y_test, y_pred, pos_label=POS_LABEL, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true_binary, y_score)),
            "report": classification_report(self.y_test, y_pred, zero_division=0),
        }

    def _log_model(self, name):
        cv = self.results[name]["cv"]
        test = self.results[name]["test"]

        self.logger.info(f"{name} best params: {cv['best_params']}")
        self.logger.info(f"{name} CV F1 folds: {cv['cv_f1_folds']}")
        self.logger.info(f"{name} CV F1 mean={cv['cv_f1_mean']:.4f} std={cv['cv_f1_std']:.4f}")
        self.logger.info(f"{name} CV ROC-AUC mean={cv['cv_roc_auc_mean']:.4f}")
        self.logger.info(f"{name} TEST confusion matrix: {test['confusion_matrix']}")
        self.logger.info(
            f"{name} TEST accuracy={test['accuracy']:.4f} precision={test['precision']:.4f} "
            f"recall={test['recall']:.4f} f1={test['f1']:.4f} roc_auc={test['roc_auc']:.4f}"
        )
        self.logger.info(f"{name} TEST report:\n{test['report']}")

    def _log_comparison(self):
        self.logger.info("Model comparison (CV F1 used for selection, TEST reported only):")
        for name in self.order:
            if name not in self.results:
                continue
            cv = self.results[name]["cv"]
            test = self.results[name]["test"]
            self.logger.info(
                f"{name:<20} CV_F1={cv['cv_f1_mean']:.4f} (std {cv['cv_f1_std']:.4f}) "
                f"TEST_F1={test['f1']:.4f} TEST_recall={test['recall']:.4f} "
                f"TEST_roc_auc={test['roc_auc']:.4f}"
            )

    def _select_champion(self):
        contenders = [n for n in self.order if n in self.results and n != BASELINE]
        if not contenders:
            raise RuntimeError("Only the baseline finished. No real model to select.")

        best_name = max(contenders, key=lambda n: self.results[n]["cv"]["cv_f1_mean"])
        best_mean = self.results[best_name]["cv"]["cv_f1_mean"]
        best_std = self.results[best_name]["cv"]["cv_f1_std"]

        # Simplest model whose CV F1 is within one std of the best one wins.
        chosen = next(
            n for n in contenders
            if self.results[n]["cv"]["cv_f1_mean"] >= best_mean - best_std
        )

        self.best_model_name = chosen
        self.best_model = self.results[chosen]["estimator"]

        self.logger.info(
            f"Champion selected: {chosen} "
            f"(CV F1 {self.results[chosen]['cv']['cv_f1_mean']:.4f})"
        )
        if chosen != best_name:
            self.logger.info(
                f"Top raw score was {best_name} ({best_mean:.4f}); gap is within noise "
                f"(std {best_std:.4f}), so the simpler model was kept."
            )

        baseline = self.results.get(BASELINE)
        if baseline and self.results[chosen]["cv"]["cv_f1_mean"] <= baseline["cv"]["cv_f1_mean"]:
            self.logger.critical("Champion does not beat the baseline. Investigate before using.")

    def _serialize_champion(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.best_model, path)
        self.logger.info(f"Champion '{self.best_model_name}' saved to {path}")
        return path