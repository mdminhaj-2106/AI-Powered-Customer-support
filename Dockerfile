FROM python:3.10-slim

WORKDIR /app


RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /tmp/transformers_cache && chmod 777 /tmp/transformers_cache


ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/transformers_cache
ENV HF_HOME=/tmp/transformers_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "app.py"]
