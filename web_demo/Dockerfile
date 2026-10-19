FROM python:3.11-slim
WORKDIR /app
COPY web_demo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt
COPY . /app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn web_demo.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
