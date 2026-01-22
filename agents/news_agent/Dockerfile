FROM python:3.13-slim

WORKDIR /app

COPY agents/news_agent /app
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb &&\
    apt install -y ./google-chrome-stable_current_amd64.deb &&\
    rm google-chrome-stable_current_amd64.deb && \
    apt-get clean
RUN pip install --no-cache-dir -r requirements.txt


# Giữ nguyên cấu trúc thư mục
COPY agents/news_agent /app