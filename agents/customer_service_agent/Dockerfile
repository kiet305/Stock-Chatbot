FROM python:3.11-slim

WORKDIR /app

COPY agents/customer_service_agent /app

RUN pip install --no-cache-dir -r requirements.txt


# Giữ nguyên cấu trúc thư mục
COPY agents/customer_service_agent /app