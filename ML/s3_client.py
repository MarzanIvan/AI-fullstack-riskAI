import boto3
import os

session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url=os.getenv("S3_ENDPOINT"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
BUCKET = os.getenv("S3_BUCKET")

def download_dataset(key: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(BUCKET, key, local_path)
    print(f"[S3] dataset downloaded: {local_path}")

def upload_model(local_path: str, key: str):
    s3.upload_file(local_path, BUCKET, key)
    print(f"[S3] model uploaded: {key}")