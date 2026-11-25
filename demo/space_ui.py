"""Presentation for the Gradio model picker, results and comparison table."""
import csv
from html import escape
from pathlib import Path

MODEL_INFO = {
    "retrained": {"title": "Fine-tuned MPNet", "csv_name": "MPNet retrained",
                  "description": "Domain-adapted sentence embeddings. Best overall test result.", "runtime": "Shared GPU"},
    "unretrained": {"title": "Base MPNet", "csv_name": "MPNet unretrained",
                    "description": "Original embeddings, without restaurant-specific fine-tuning.", "runtime": "Shared GPU"},
    "tfidf": {"title": "TF-IDF", "csv_name": "TF-IDF",
              "description": "A lightweight lexical baseline. No GPU quota needed.", "runtime": "CPU · no GPU quota"},
}
DEFAULT_MODEL = "retrained"
with (Path(__file__).resolve().parents[1] / "outputs/final_comparison/final_model_comparison.csv").open(encoding="utf-8") as f:
    METRICS = {row["model"]: row for row in csv.DictReader(f)}

HEADER = """
<header class="app-header">
  <div class="brand"><span class="brand-icon" aria-hidden="true"><i></i><i></i><i></i></span>
    <span>AspectAI</span><span class="brand-caption">Restaurant review analysis</span></div>
  <a class="github-link" href="https://github.com/zaabinh/CS221-Aspect-Classification"
     target="_blank" rel="noopener noreferrer">View on GitHub ↗</a>
</header>
<section class="hero"><div class="eyebrow">ONE REVIEW. MORE THAN ONE ASPECT.</div>
  <h1>Which parts of the meal<br>is this review about?</h1>
  <p>Great food, slow service, or a little of both? Paste an English review and see which
     of the five aspects it mentions. Switch models to compare their predictions.</p>
</section>
"""
EMPTY_RESULTS = """
<section class="analysis-result" aria-live="polite">
  <div class="eyebrow">02 / THE BREAKDOWN</div><h2>Aspect predictions</h2>
  <div class="empty-state"><div class="empty-symbol" aria-hidden="true">≋</div>
    <h3>A review has more than one side.</h3>
    <p>Your predicted aspects and model scores will appear here.</p>
    <div class="empty-aspects"><span>Food</span><span>Price</span><span>Service</span>
      <span>Ambiance</span><span>Miscellaneous</span></div></div>
  <p class="score-disclaimer">Model scores describe the classifier's output, not calibrated probabilities.</p>
</section>
"""


def render_model_card(model_key):
    info = MODEL_INFO[model_key]
    metrics = METRICS[info["csv_name"]]
    return (
        f'<div class="model-summary"><p>{escape(info["description"])}</p>'
        f'<div class="model-facts"><span><b>{float(metrics["macro_f1"]):.3f}</b> Macro F1</span>'
        f'<span><b>{float(metrics["micro_f1"]):.3f}</b> Micro F1</span>'
        f'<span class="runtime-badge">{escape(info["runtime"])}</span></div></div>'
    )


def render_result(result, model_key):
    title = escape(MODEL_INFO[model_key]["title"])
    tags = "".join(
        f'<span class="aspect-tag" data-aspect="{escape(a, quote=True)}">{escape(a.title())}</span>'
        for a in result["aspects"]
    )
    bars = []
    for item in result["scores"]:
        aspect = escape(item["aspect"], quote=True)
        score, threshold = float(item["score"]), float(item["threshold"])
        width, marker = max(0, min(100, score * 100)), max(0, min(100, threshold * 100))
        selected = "selected" if item["predicted"] else "unselected"
        bars.append(
            f'<div class="aspect-row {selected}" data-aspect="{aspect}">'
            f'<div class="aspect-row-head"><span>{escape(item["aspect"].title())}</span>'
            f'<span class="score-value">{score:.4f}</span></div>'
            f'<div class="score-track" role="meter" aria-label="{aspect} model score" '
            f'aria-valuemin="0" aria-valuemax="1" aria-valuenow="{score}">'
            f'<span class="score-fill" style="width:{width:.2f}%"></span>'
            f'<span class="threshold-marker" style="left:{marker:.2f}%"></span></div>'
            f'<div class="aspect-row-foot"><span>{"Selected" if item["predicted"] else "Below cutoff"}</span>'
            f'<span>Cutoff {threshold:.2f}</span></div></div>'
        )
    fallback = '<p class="fallback-note">No cutoff passed; the highest-scoring aspect was selected.</p>' if result["fallback_used"] else ""
    return (
        '<section class="analysis-result" aria-live="polite"><div class="eyebrow">02 / THE BREAKDOWN</div>'
        f'<div class="result-heading"><h2>Aspect predictions</h2><span>{len(result["aspects"])} selected</span></div>'
        f'<div class="aspect-tags">{tags}</div>{fallback}<div class="score-list">{"".join(bars)}</div>'
        '<p class="score-disclaimer">The marker is the selection cutoff. Scores are not calibrated probabilities.</p>'
        f'<div class="result-footer"><span>{title}</span>'
        f'<span>{float(result["latency_ms"]):,.1f} ms · excludes queue time</span></div></section>'
    )


def render_evaluation():
    rows = []
    for key, info in MODEL_INFO.items():
        metrics = METRICS[info["csv_name"]]
        badge = '<span class="best-badge">Best overall</span>' if key == DEFAULT_MODEL else ""
        cells = "".join(f'<td>{float(metrics[c]):.3f}</td>' for c in ["macro_f1", "micro_f1", "samples_f1", "exact_match_ratio", "hamming_loss"])
        rows.append(f'<tr><th scope="row">{escape(info["title"])}{badge}</th>{cells}</tr>')
    return (
        '<section class="evaluation"><div class="eyebrow">THE BENCHMARK</div><h2>Three approaches. The same test set.</h2>'
        '<p>160 held-out reviews. Cutoffs tuned on validation data. Higher is better except Hamming loss.</p>'
        '<div class="evaluation-scroll"><table><thead><tr><th scope="col">Model</th><th scope="col">Macro F1</th>'
        '<th scope="col">Micro F1</th><th scope="col">Samples F1</th><th scope="col">Exact match</th>'
        f'<th scope="col">Hamming loss ↓</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="evaluation-note">The miscellaneous category is less reliable: F1 0.20 with fine-tuned MPNet. '
        'Small wording changes can change a prediction.</p></section>'
    )
