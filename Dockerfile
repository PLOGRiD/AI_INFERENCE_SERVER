FROM python:3.12-slim

WORKDIR /app

# ultralytics(OpenCV)가 필요로 하는 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py pipeline.py ./
COPY yolo_app ./yolo_app
COPY nir_app ./nir_app

ENV YOLO_MODEL_DIR=/app/models
ENV NIR_MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
