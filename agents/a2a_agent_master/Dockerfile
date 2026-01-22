FROM python:3.13-slim

WORKDIR /app

COPY agents/a2a_agent_master /app

RUN pip install --no-cache-dir -r requirements.txt


# Giữ nguyên cấu trúc thư mục
COPY agents/a2a_agent_master /app