"""Check the Space boundary without requesting GPU time or importing its UI stack."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nltk
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))


class UserError(Exception):
    pass


@pytest.fixture
def space_app(monkeypatch):
    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace(GPU=lambda **_: lambda fn: fn))
    monkeypatch.setitem(sys.modules, "gradio", SimpleNamespace(Error=UserError))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(nltk, "download", Mock())
    spec = importlib.util.spec_from_file_location("aspectai_space_test", DEMO_DIR / "space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.predict_on_gpu = Mock()
    return module


@pytest.mark.parametrize("text", [None, 42, "", "  ", "!!!", "a" * 1001])
def test_invalid_reviews_do_not_request_gpu(space_app, text):
    with pytest.raises(UserError):
        space_app.analyze(text)
    space_app.predict_on_gpu.assert_not_called()


def test_space_preserves_multiple_labels_and_threshold_details(space_app):
    space_app.predict_on_gpu.return_value = {
        "aspects": ["food", "price"],
        "scores": [
            {"aspect": "food", "score": 0.8, "threshold": 0.43, "predicted": True},
            {"aspect": "price", "score": 0.6, "threshold": 0.44, "predicted": True},
        ],
        "latency_ms": 12.5,
        "fallback_used": False,
    }
    aspects, rows, note = space_app.analyze("  tasty but expensive  ")
    space_app.predict_on_gpu.assert_called_once_with("tasty but expensive")
    assert aspects == "food, price"
    assert rows == [["food", 0.8, 0.43, True], ["price", 0.6, 0.44, True]]
    assert "12.5 ms" in note


def test_space_hides_internal_prediction_errors(space_app):
    space_app.predict_on_gpu.side_effect = RuntimeError("internal model path")
    with pytest.raises(UserError) as error:
        space_app.analyze("good food")
    assert "internal model path" not in str(error.value)


def test_zerogpu_model_loading_does_not_execute_warmup(monkeypatch, tmp_path):
    from inference import AspectClassifier

    bundle = tmp_path / "classifiers" / "retrained"
    encoder_dir = tmp_path / "retrained_encoder"
    bundle.mkdir(parents=True)
    encoder_dir.mkdir()
    (bundle / "metadata.json").write_text(json.dumps({
        "aspect_cols": ["food"], "thresholds": {"food": 0.43},
        "encoder_path": "models/mpnet/retrained_encoder",
    }))
    (bundle / "model.joblib").touch()
    (encoder_dir / "model.safetensors").touch()
    encoder = Mock()
    factory = Mock(return_value=encoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    monkeypatch.setattr("inference.joblib.load", Mock())

    model = AspectClassifier(tmp_path, device="cuda", warmup=False)

    factory.assert_called_once_with(str(encoder_dir), device="cuda")
    assert model.encoder is encoder
    encoder.encode.assert_not_called()
