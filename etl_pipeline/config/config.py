import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()


# MinIO config
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT"),
    "bucket_name": os.getenv("DATALAKE_BUCKET"),
    "access_key": os.getenv("MINIO_ACCESS_KEY"),
    "secret_key": os.getenv("MINIO_SECRET_KEY")
}

# PostgreSQL config
PSQL_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")),
    "database": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

print(MINIO_CONFIG)