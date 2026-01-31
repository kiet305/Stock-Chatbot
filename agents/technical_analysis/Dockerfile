FROM python:3.11-slim

WORKDIR /app

# Copy requirements trước để cache tốt + install đúng
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source sau
COPY . /app

CMD ["python", "main.py"]