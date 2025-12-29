import boto3
import os
# from utils.utllity import (
#     AWS_ACCESS_KEY_ID,
#     AWS_SECRET_ACCESS_KEY,
#     AWS_REGION,
#     S3_BUCKET,
# )


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
    )

def download_file(s3_uri, local_path):
    if s3_uri.startswith("http"):
        import requests
        r = requests.get(s3_uri, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return

    # s3://bucket/key
    _, _, bucket, *key = s3_uri.split("/")
    key = "/".join(key)

    s3 = get_s3_client()
    s3.download_file(bucket, key, local_path)

def upload_file(local_path, key):
    s3 = get_s3_client()
    bucket = os.environ["LAMBDA_BUCKET"]
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"
