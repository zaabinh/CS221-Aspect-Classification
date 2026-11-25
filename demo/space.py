"""Gradio entry point for the free Hugging Face ZeroGPU Space."""
import spaces  # Must precede torch so ZeroGPU can initialize its CUDA hooks.

import logging
from pathlib import Path

import gradio as gr
import nltk
import torch

for resource in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(resource, quiet=True, raise_on_error=True)

from inference import MAX_REVIEW_CHARS, AspectClassifier, TfidfAspectClassifier  # noqa: E402
from space_ui import (  # noqa: E402
    DEFAULT_MODEL, EMPTY_RESULTS, HEADER, MODEL_INFO, render_evaluation,
    render_model_card, render_result,
)

log = logging.getLogger("aspectai.space")
models = {}
SAMPLES = [
    ("Great food, high price", "The lamb biryani was delicious and the portions were huge, but it is far too expensive for what you get."),
    ("Rude staff, nice room", "Our waiter was rude and ignored us for twenty minutes, even though the dining room is beautifully decorated."),
    ("Cheap, friendly, cozy", "Cheap prices, friendly staff and a cozy, quiet atmosphere. We will definitely come back."),
    ("Location and parking", "The restaurant is hard to find and there is nowhere to park nearby."),
]


@spaces.GPU(duration=10)
def predict_on_gpu(text, model_key):
    return models[model_key].predict(text)


def analyze(text, model_key=DEFAULT_MODEL):
    if not isinstance(model_key, str) or model_key not in MODEL_INFO:
        raise gr.Error("Please select one of the available models.")
    if not isinstance(text, str) or not text.strip():
        raise gr.Error("Please enter a review.")
    text = text.strip()
    if len(text) > MAX_REVIEW_CHARS:
        raise gr.Error(f"Review is too long (max {MAX_REVIEW_CHARS} characters).")
    if not any(c.isalpha() and c.isascii() for c in text):
        raise gr.Error("Please enter an English review with usable words.")
    try:
        result = models[model_key].predict(text) if model_key == "tfidf" else predict_on_gpu(text, model_key)
    except ValueError as error:
        raise gr.Error(str(error)) from None
    except Exception:
        log.exception("Prediction failed for %s", model_key)
        raise gr.Error("Prediction is unavailable. For MPNet, check your GPU quota or try TF-IDF.") from None
    rows = [[s["aspect"], s["score"], s["threshold"], s["predicted"]] for s in result["scores"]]
    note = f"Inference: {result['latency_ms']} ms (excludes queue time)."
    if result["fallback_used"]:
        note += " No threshold passed; the highest-scoring aspect was selected."
    return ", ".join(result["aspects"]), rows, note, render_result(result, model_key)


def create_demo():
    with gr.Blocks(title="AspectAI · Restaurant Review Analysis", analytics_enabled=False) as demo:
        gr.HTML(HEADER, elem_id="app-intro")
        with gr.Row(elem_id="workbench", equal_height=True):
            with gr.Column(scale=1, min_width=320, elem_id="input-panel"):
                gr.HTML('<div class="eyebrow">01 / YOUR REVIEW</div><h2 class="panel-title">Try a different perspective.</h2>')
                selector = gr.Radio(
                    choices=[(info["title"], key) for key, info in MODEL_INFO.items()],
                    value=DEFAULT_MODEL, label="Choose a model", elem_id="model-picker",
                )
                summary = gr.HTML(render_model_card(DEFAULT_MODEL), elem_id="model-summary")
                review = gr.Textbox(
                    label="Restaurant review", lines=6, max_length=MAX_REVIEW_CHARS,
                    placeholder="The curry was rich and full of flavour, but we waited forty minutes for a table…",
                    elem_id="review-input",
                )
                counter = gr.HTML('<span>0 / 1,000 characters</span>', elem_id="char-counter")
                with gr.Row(elem_id="review-actions"):
                    clear = gr.Button("Clear", variant="secondary", min_width=80, elem_id="clear-review")
                    submit = gr.Button("Analyze review →", variant="primary", elem_id="analyze-review")
                gr.HTML('<div class="samples-title">Or try a sample</div>')
                with gr.Row(elem_id="sample-buttons"):
                    for title, sample in SAMPLES:
                        button = gr.Button(title, size="sm", min_width=150)
                        button.click(lambda value=sample: value, outputs=review, queue=False, api_name=False)
            with gr.Column(scale=1, min_width=320, elem_id="result-panel"):
                result = gr.HTML(EMPTY_RESULTS, elem_id="prediction-result")
                # Keep structured API outputs alongside the visual results.
                labels = gr.Textbox(visible=False)
                scores = gr.Dataframe(
                    headers=["Aspect", "Model score", "Threshold", "Selected"],
                    datatype=["str", "number", "number", "bool"], visible=False,
                )
                details = gr.Textbox(visible=False)
        gr.HTML('<p class="hosting-note">MPNet uses a shared GPU and may queue. TF-IDF runs on CPU without GPU quota. English reviews only.</p>')
        gr.HTML(render_evaluation(), elem_id="model-evaluation")
        gr.HTML('<footer class="app-footer"><span>AspectAI · CS221 Natural Language Processing</span><span>University of Information Technology, VNU-HCM</span></footer>')
        submit.click(analyze, [review, selector], [labels, scores, details, result], api_name="analyze")
        selector.change(
            lambda key: (render_model_card(key), EMPTY_RESULTS), selector, [summary, result],
            queue=False, api_name=False,
        )
        clear.click(lambda: ("", EMPTY_RESULTS), outputs=[review, result], queue=False, api_name=False)
        review.change(
            fn=None, inputs=review, outputs=counter,
            js="(text) => `<span>${text.length.toLocaleString('en-US')} / 1,000 characters</span>`",
            queue=False, api_name=False,
        )
    return demo


if __name__ == "__main__":
    # CUDA weights load at module scope; only decorated calls execute GPU inference.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {key: AspectClassifier(classifier_name=key, device=device, warmup=False) for key in ("retrained", "unretrained")}
    models["tfidf"] = TfidfAspectClassifier()
    css = Path(__file__).with_name("space.css").read_text(encoding="utf-8")
    theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate", font=["Inter", "system-ui", "sans-serif"])
    create_demo().queue(default_concurrency_limit=1, max_size=20).launch(show_error=False, css=css, theme=theme)
