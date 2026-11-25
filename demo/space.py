"""Gradio entry point for the free Hugging Face ZeroGPU Space."""
import spaces  # Must be imported before torch so ZeroGPU can initialize its CUDA hooks.

import logging

import gradio as gr
import nltk
import torch

nltk.download("stopwords", quiet=True, raise_on_error=True)

from inference import MAX_REVIEW_CHARS, AspectClassifier  # noqa: E402

log = logging.getLogger("aspectai.space")
model = None


@spaces.GPU(duration=10)
def predict_on_gpu(text):
    return model.predict(text)


def analyze(text):
    if not isinstance(text, str) or not text.strip():
        raise gr.Error("Please enter a review.")
    text = text.strip()
    if len(text) > MAX_REVIEW_CHARS:
        raise gr.Error(f"Review is too long (max {MAX_REVIEW_CHARS} characters).")
    if not any(c.isalpha() and c.isascii() for c in text):
        raise gr.Error("Please enter an English review with usable words.")
    try:
        result = predict_on_gpu(text)
    except ValueError as error:
        raise gr.Error(str(error)) from None
    except Exception:
        log.exception("Prediction failed")
        raise gr.Error("Prediction is unavailable. Check your GPU quota or try again shortly.") from None
    rows = [
        [s["aspect"], s["score"], s["threshold"], s["predicted"]]
        for s in result["scores"]
    ]
    note = f"Inference: {result['latency_ms']} ms (excludes queue time)."
    if result["fallback_used"]:
        note += " No threshold passed; the highest-scoring aspect was selected."
    return ", ".join(result["aspects"]), rows, note


def create_demo():
    return gr.Interface(
        fn=analyze,
        inputs=gr.Textbox(label="Restaurant review", lines=4, max_length=MAX_REVIEW_CHARS),
        outputs=[
            gr.Textbox(label="Predicted aspects"),
            gr.Dataframe(
                headers=["Aspect", "Model score", "Threshold", "Selected"],
                datatype=["str", "number", "number", "bool"],
                interactive=False,
            ),
            gr.Textbox(label="Prediction details"),
        ],
        title="AspectAI — Restaurant Review Aspects",
        description=(
            "Identify food, price, service, ambiance and miscellaneous in an English review. "
            "Fine-tuned MPNet + manual logistic regression. Scores are not calibrated probabilities. "
            "This free demo uses shared GPU capacity; requests can queue or reach a daily quota."
        ),
        article=(
            "Test-set macro F1: 0.812; micro F1: 0.909 (160 reviews). "
            "The miscellaneous aspect is less reliable (F1: 0.20). "
            "[Source and model limitations](https://github.com/zaabinh/CS221-Aspect-Classification)."
        ),
        examples=[
            ["The food was tasty but the price was high."],
            ["Friendly staff and a lovely atmosphere."],
        ],
        cache_examples=False,
        flagging_mode="never",
        analytics_enabled=False,
    )


if __name__ == "__main__":
    # Load CUDA weights at module scope; only decorated calls may execute GPU inference.
    model = AspectClassifier(device="cuda" if torch.cuda.is_available() else "cpu", warmup=False)
    create_demo().queue(default_concurrency_limit=1, max_size=20).launch(show_error=False)
