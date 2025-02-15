FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libxrender-dev

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "vehicle-counter-zone.py"]
