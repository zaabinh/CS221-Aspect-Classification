import csv
import logging
import os
import threading
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request

from inference import MAX_REVIEW_CHARS, PROJECT_ROOT, AspectClassifier

log = logging.getLogger("aspectai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GITHUB_URL = "https://github.com/zaabinh/CS221-Aspect-Classification"
RESULTS_CSV = PROJECT_ROOT / "outputs" / "final_comparison" / "final_model_comparison.csv"
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))


def load_results():
    """Test-set metrics as committed by notebooks/final_comparison.ipynb (160 held-out reviews)."""
    try:
        with RESULTS_CSV.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        log.warning("Results file not found: %s", RESULTS_CSV)
        return []


class RateLimiter:
    # shortcut: per-process memory, resets on restart and is not shared across workers; use Redis if scaled out.
    def __init__(self, limit, window=60.0):
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self.hits) > 10_000:  # drop idle clients so memory stays bounded
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window]:
                    del self.hits[k]
            return True


def create_app(model=None, load_model=True):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    limiter = RateLimiter(RATE_LIMIT)
    load_error = None

    if model is None and load_model:
        try:
            model = AspectClassifier()
            log.info("Model loaded: %s, aspects=%s", model.encoder_name, model.aspects)
        except Exception as error:  # keep serving the page; /analyze reports the model as unavailable
            load_error = str(error)
            log.exception("Model failed to load")

    results = load_results()

    def client_ip():
        # shortcut: first X-Forwarded-For entry is client-controlled, so the limit is a soft guard against
        # accidental floods, not abuse; put a proxy-level limit in front if the demo gets real traffic.
        return (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model=model,
            load_error=load_error,
            results=results,
            github_url=GITHUB_URL,
            max_chars=MAX_REVIEW_CHARS,
        )

    @app.get("/healthz")
    def healthz():
        if model is None:
            return jsonify(status="error", model_loaded=False), 503
        return jsonify(status="ok", model_loaded=True)

    @app.post("/analyze")
    def analyze():
        if not limiter.allow(client_ip()):
            return jsonify(error="Too many requests. Please wait a minute and try again."), 429

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            return jsonify(error='Send JSON like {"text": "your review"}.'), 400

        text = data["text"].strip()
        if not text:
            return jsonify(error="Please enter a review."), 400
        if len(text) > MAX_REVIEW_CHARS:
            return jsonify(error=f"Review is too long (max {MAX_REVIEW_CHARS} characters)."), 400
        if model is None:
            return jsonify(error="The model is not available on this server right now."), 503

        try:
            return jsonify(model.predict(text))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            log.exception("Inference failed")  # review text is deliberately not logged
            return jsonify(error="Something went wrong while analyzing the review."), 500

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Request is too large."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
