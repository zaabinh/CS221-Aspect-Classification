import csv
import sys
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

from app import create_app  # noqa: E402
from inference import MAX_REVIEW_CHARS, MODEL_DIR, PROJECT_ROOT, AspectClassifier, apply_thresholds  # noqa: E402
from preprocessing import clean_review_text, preprocess_review  # noqa: E402

ASPECTS = ["food", "price", "service", "ambiance", "miscellaneous"]
# Git tracks the folder as "Data/"; notebooks read "data/" (same folder on Windows, not on Linux).
DATA_CSV = next(
    (p for p in (PROJECT_ROOT / d / "Restaurant_ABSA_processed.csv" for d in ("data", "Data")) if p.exists()),
    None,
)
HAS_MODEL = (MODEL_DIR / "retrained_encoder" / "model.safetensors").exists()


# ---------- preprocessing ----------

@pytest.mark.skipif(DATA_CSV is None, reason="processed dataset not available")
def test_preprocessing_matches_training_column():
    with DATA_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # review_en is already the cleaned form, so cleaning must be a no-op on it ...
    assert [r["review_en"] for r in rows if clean_review_text(r["review_en"]) != r["review_en"]] == []
    # ... and the ML form must match the column the classifier was fit on.
    mismatches = [r for r in rows if preprocess_review(r["review_en"]) != r["review_cleaned"]]
    assert not mismatches, f"{len(mismatches)} rows differ, e.g. {mismatches[0]}"


def test_cleaning_matches_notebook_rules():
    assert clean_review_text("  The Food   wasn't GREAT!!​ ") == "the food was not great"


def test_preprocessing_keeps_negations():
    assert preprocess_review("The food wasn't good!") == "food not good"


@pytest.mark.parametrize("text", ["", "   ", "!!! ???"])
def test_preprocessing_rejects_empty(text):
    with pytest.raises(ValueError):
        preprocess_review(text)


# ---------- thresholding ----------

THRESHOLDS = {"food": 0.43, "price": 0.44, "service": 0.52, "ambiance": 0.52, "miscellaneous": 0.5}


def test_thresholds_allow_multiple_labels():
    predicted, fallback = apply_thresholds(np.array([0.9, 0.44, 0.51, 0.8, 0.1]), THRESHOLDS, ASPECTS)
    assert predicted.tolist() == [True, True, False, True, False]  # price == threshold counts as predicted
    assert not fallback


def test_thresholds_fallback_to_top_aspect():
    predicted, fallback = apply_thresholds(np.array([0.1, 0.2, 0.3, 0.45, 0.05]), THRESHOLDS, ASPECTS)
    assert predicted.tolist() == [False, False, False, True, False]
    assert fallback


# ---------- real model ----------

@pytest.fixture(scope="module", params=["retrained", "unretrained"])
def model(request):
    if not (MODEL_DIR / f"{request.param}_encoder" / "model.safetensors").exists():
        pytest.skip("model artifacts not available (git lfs pull)")
    return AspectClassifier(classifier_name=request.param)


def test_model_loads_with_training_label_order(model):
    assert model.aspects == ASPECTS
    assert set(model.thresholds) == set(ASPECTS)
    assert len(model.classifier.estimators_) == len(ASPECTS)


def test_prediction_structure_and_multilabel(model):
    out = model.predict("The food was tasty but the price was high.")
    assert set(out) == {"aspects", "scores", "fallback_used", "processed_text", "latency_ms"}
    assert [s["aspect"] for s in out["scores"]] == ASPECTS
    for s in out["scores"]:
        assert 0.0 <= s["score"] <= 1.0
        assert s["predicted"] == (s["score"] >= s["threshold"]) or out["fallback_used"]
    assert out["aspects"] == [s["aspect"] for s in out["scores"] if s["predicted"]]
    assert {"food", "price"} <= set(out["aspects"])


@pytest.mark.skipif(DATA_CSV is None, reason="processed dataset not available")
def test_served_pipeline_reproduces_published_test_metrics(model):
    """The demo must score exactly what outputs/final_comparison/final_model_comparison.csv reports."""
    from sklearn.metrics import f1_score, hamming_loss

    with DATA_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    test_idx = np.load(PROJECT_ROOT / "outputs" / "MPNet" / "data_split.npz")["test_idx"]
    texts = [model.transform(rows[i]["review_en"]) for i in test_idx]
    truth = np.array([[int(rows[i][a]) for a in ASPECTS] for i in test_idx])

    scores = model.classifier.predict_proba(model.encoder.encode(texts, normalize_embeddings=True))
    preds = np.array([apply_thresholds(s, model.thresholds, model.aspects)[0] for s in scores]).astype(int)

    with (PROJECT_ROOT / "outputs" / "final_comparison" / "final_model_comparison.csv").open(encoding="utf-8") as f:
        published = next(r for r in csv.DictReader(f) if r["model"] == f"MPNet {model.metadata['encoder_name']}")
    assert f1_score(truth, preds, average="macro") == pytest.approx(float(published["macro_f1"]), abs=1e-6)
    assert f1_score(truth, preds, average="micro") == pytest.approx(float(published["micro_f1"]), abs=1e-6)
    assert hamming_loss(truth, preds) == pytest.approx(float(published["hamming_loss"]), abs=1e-6)


# ---------- HTTP layer (fake model, no heavy load) ----------

class FakeModel:
    encoder_name = "fake"
    aspects = ASPECTS
    thresholds = THRESHOLDS
    calls = 0

    def predict(self, text):
        FakeModel.calls += 1
        if text == "boom":
            raise RuntimeError("secret internal detail")
        return {"aspects": ["food"], "scores": [], "fallback_used": False, "processed_text": text, "latency_ms": 1.0}


@pytest.fixture
def client():
    return create_app(model=FakeModel()).test_client()


def test_index_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/healthz").get_json() == {"status": "ok", "model_loaded": True}


def test_analyze_ok(client):
    r = client.post("/analyze", json={"text": "  tasty food  "})
    assert r.status_code == 200 and r.get_json()["aspects"] == ["food"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json": {"text": ""}},
        {"json": {"text": "   "}},
        {"json": {}},
        {"json": {"text": 42}},
        {"json": ["text"]},
        {"data": "not json", "content_type": "application/json"},
        {"json": {"text": "a" * (MAX_REVIEW_CHARS + 1)}},
    ],
)
def test_analyze_rejects_invalid_input(client, kwargs):
    r = client.post("/analyze", **kwargs)
    assert r.status_code == 400 and "error" in r.get_json()


def test_analyze_hides_internal_errors(client):
    r = client.post("/analyze", json={"text": "boom"})
    assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)


def test_request_size_limit(client):
    r = client.post("/analyze", data="x" * 20_000, content_type="application/json")
    assert r.status_code == 413


def test_model_unavailable():
    c = create_app(model=None, load_model=False).test_client()
    assert c.get("/healthz").status_code == 503
    assert c.post("/analyze", json={"text": "good food"}).status_code == 503
    assert c.get("/").status_code == 200


def test_rate_limit(monkeypatch):
    monkeypatch.setattr("app.RATE_LIMIT", 2)
    c = create_app(model=FakeModel()).test_client()
    codes = [c.post("/analyze", json={"text": "food"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
