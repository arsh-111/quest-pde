# Hugging Face Space (Docker SDK). CPU-only PyTorch; serves FastAPI on port 7860.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.space.txt .
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.space.txt

COPY --chown=user . .
USER user
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
