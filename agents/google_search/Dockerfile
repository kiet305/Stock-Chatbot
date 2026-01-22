FROM python:3.13-slim

WORKDIR /app

COPY agents/google_search /app

RUN pip install --no-cache-dir -r requirements.txt


# Giữ nguyên cấu trúc thư mục
COPY agents/google_search /app