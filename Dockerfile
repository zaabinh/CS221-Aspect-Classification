# AspectAI demo image (Hugging Face Spaces, Docker SDK). Build from the repository root.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/opt/nltk_data \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    PORT=7860

WORKDIR /app

# CPU-only torch first so the requirements step does not pull CUDA wheels.
COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 \
 && pip install --no-cache-dir -r demo/requirements.txt \
 && python -m nltk.downloader -q -d /opt/nltk_data stopwords wordnet omw-1.4

# Only what inference needs: ~420 MB encoder + classifier bundle (Git LFS files must be pulled, not pointers).
COPY preprocessing.py manual_ovr_logistic_regression.py ./
COPY models/mpnet/retrained_encoder models/mpnet/retrained_encoder
COPY models/mpnet/classifiers/retrained models/mpnet/classifiers/retrained
COPY outputs/final_comparison/final_model_comparison.csv outputs/final_comparison/
COPY demo/app.py demo/inference.py demo/
COPY demo/templates demo/templates
COPY demo/static demo/static

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"

# One worker so the model is loaded once; threads serve concurrent requests.
CMD gunicorn --chdir demo --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 "app:create_app()"
