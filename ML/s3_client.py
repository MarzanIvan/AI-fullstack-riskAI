import os
import io
import boto3
import pandas as pd
from botocore.client import Config
from typing import Optional


class S3Client:
    def __init__(
        self,
        bucket: str,
        endpoint: Optional[str] = "https://storage.yandexcloud.net",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.bucket = bucket

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=aws_access_key_id or os.getenv("YANDEX_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("YANDEX_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="ru-central1"
        )
    def read_csv(
        self,
        key: str,
        sep: str = ";",
        encoding: str = "utf-8"
    ) -> pd.DataFrame:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        df = pd.read_csv(
            io.BytesIO(obj["Body"].read()),
            sep=sep,
            encoding=encoding,
            engine="python",
            skipinitialspace=True
        )
        df.columns = df.columns.str.strip()
        return df


    def read_json(self, key: str) -> pd.DataFrame:
        """Read JSON file """
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_json(io.BytesIO(obj["Body"].read()))

    def read_parquet(self, key: str) -> pd.DataFrame:
        """Read PARQUET file """
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def download_model(self, key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path)
        return local_path

    def upload_model(self, local_path: str, key: str):
        self.s3.upload_file(local_path, self.bucket, key)

    def upload_file(self, local_path: str, key: str):
        """Upload any file to S3."""
        self.s3.upload_file(local_path, self.bucket, key)

    def download_file(self, key: str, local_path: str):
        """Download any file """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path)

    def list_files(self, prefix: str = ""):
        """List objects in bucket."""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [item["Key"] for item in response["Contents"]]

