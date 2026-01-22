FROM python:3.13-slim

WORKDIR /app

COPY agents/company_search_agent /app

RUN pip install --no-cache-dir -r requirements.txt


# Giữ nguyên cấu trúc thư mục
COPY agents/company_search_agent /app