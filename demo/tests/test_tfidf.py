"""Check the sparse serving pipeline against the published held-out benchmark."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, hamming_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import PROJECT_ROOT, TfidfAspectClassifier, apply_thresholds  # noqa: E402


def test_tfidf_serving_reproduces_published_metrics():
    data = next((PROJECT_ROOT / d / "Restaurant_ABSA_processed.csv" for d in ("data", "Data")
                 if (PROJECT_ROOT / d / "Restaurant_ABSA_processed.csv").exists()), None)
    if data is None or not (PROJECT_ROOT / "models/TF-IDF/vectorizer.joblib").exists():
        pytest.skip("local dataset/model artifacts not available")
    model = TfidfAspectClassifier()
    with data.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    test_idx = np.load(PROJECT_ROOT / "outputs/MPNet/data_split.npz")["test_idx"]
    predictions = np.array([
        apply_thresholds(model._predict_scores(model.transform(rows[i]["review_en"])), model.thresholds, model.aspects)[0]
        for i in test_idx
    ])
    truth = np.array([[int(rows[i][a]) for a in model.aspects] for i in test_idx])
    with (PROJECT_ROOT / "outputs/final_comparison/final_model_comparison.csv").open(encoding="utf-8") as f:
        published = next(r for r in csv.DictReader(f) if r["model"] == "TF-IDF")
    assert f1_score(truth, predictions, average="macro") == pytest.approx(float(published["macro_f1"]), abs=1e-6)
    assert f1_score(truth, predictions, average="micro") == pytest.approx(float(published["micro_f1"]), abs=1e-6)
    assert hamming_loss(truth, predictions) == pytest.approx(float(published["hamming_loss"]), abs=1e-6)
    output = model.predict("The food was tasty but the price was high.")
    assert [s["aspect"] for s in output["scores"]] == ["food", "price", "service", "ambiance", "miscellaneous"]
