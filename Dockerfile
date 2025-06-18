FROM python:3.9-slim
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir AMT-Hypervector
ENTRYPOINT ["bhre"]
