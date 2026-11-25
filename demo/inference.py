"""Aspect classifier inference: preprocessing -> MPNet embedding -> OvR logistic regression -> thresholds."""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# model.joblib pickles manual_ovr_logistic_regression.ManualLogisticRegression, so the root must be importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import clean_review_text  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models" / "mpnet"))
CLASSIFIER_NAME = os.environ.get("CLASSIFIER_NAME", "retrained")
MAX_REVIEW_CHARS = 1000


class ModelNotLoadedError(RuntimeError):
    pass


def apply_thresholds(scores: np.ndarray, thresholds: dict, aspects: list) -> tuple[np.ndarray, bool]:
    """Same rule as the training notebooks: score >= per-aspect threshold; if nothing passes, keep the top aspect."""
    predicted = np.array([scores[i] >= thresholds[a] for i, a in enumerate(aspects)])
    fallback = not predicted.any()
    if fallback:
        predicted[int(np.argmax(scores))] = True
    return predicted, fallback


class AspectClassifier:
    def __init__(
        self, model_dir: Path = MODEL_DIR, classifier_name: str = CLASSIFIER_NAME,
        *, device: str = "cpu", warmup: bool = True,
    ):
        bundle = Path(model_dir) / "classifiers" / classifier_name
        with (bundle / "metadata.json").open(encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.aspects = self.metadata["aspect_cols"]
        self.thresholds = self.metadata["thresholds"]
        # metadata["text_col"] says the classifier was fit on `review_cleaned`, but the committed test metrics
        # (03-mpnet_evaluate, final_comparison) feed `review_en`. Serve what was evaluated so the published
        # numbers describe this demo; see README "Limitations".
        self.transform = clean_review_text

        # metadata stores a Windows-style path relative to the project root; only the folder name is needed.
        encoder_dir = Path(model_dir) / Path(self.metadata["encoder_path"].replace("\\", "/")).name
        missing = [p for p in (bundle / "model.joblib", encoder_dir / "model.safetensors") if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing model artifacts: " + ", ".join(map(str, missing)))

        from sentence_transformers import SentenceTransformer  # heavy import, deferred until a model is actually loaded

        self.classifier = joblib.load(bundle / "model.joblib")
        self.encoder = SentenceTransformer(str(encoder_dir), device=device)
        self.encoder_name = encoder_dir.name
        if warmup:
            self.encoder.encode(["warm up"])  # ZeroGPU inference must wait for a decorated call.

    def predict(self, review: str) -> dict:
        start = time.perf_counter()
        processed = self.transform(review)
        processed = processed if any(c.isalpha() for c in processed) else ""
        if not processed:
            raise ValueError("Review has no usable words after preprocessing.")

        embedding = self.encoder.encode(
            [processed],
            normalize_embeddings=self.metadata.get("normalize_embeddings", True),
            convert_to_numpy=True,
        )
        scores = self.classifier.predict_proba(embedding)[0]
        predicted, fallback = apply_thresholds(scores, self.thresholds, self.aspects)

        return {
            "aspects": [a for a, p in zip(self.aspects, predicted) if p],
            "scores": [
                {
                    "aspect": a,
                    "score": round(float(s), 4),
                    "threshold": self.thresholds[a],
                    "predicted": bool(p),
                }
                for a, s, p in zip(self.aspects, scores, predicted)
            ],
            "fallback_used": fallback,
            "processed_text": processed,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
