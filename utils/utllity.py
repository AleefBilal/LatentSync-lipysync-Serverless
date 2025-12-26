import os
import cv2
from dotenv import load_dotenv, find_dotenv

# ============================
# Runtime globals (per request)
# ============================
RUNTIME_ENV = None
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None
AWS_REGION = None
S3_BUCKET = None


def reset_runtime_env():
    global RUNTIME_ENV
    global AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET

    RUNTIME_ENV = None
    AWS_ACCESS_KEY_ID = None
    AWS_SECRET_ACCESS_KEY = None
    AWS_REGION = None
    S3_BUCKET = None


def load_environment(env_key: str = "stag"):
    """
    Configure runtime environment (stag / prod).

    - On RunPod: uses injected env vars
    - Locally: loads stag.env / prod.env if present
    """
    global RUNTIME_ENV
    global AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET

    if env_key not in ("stag", "prod"):
        raise ValueError("env_key must be 'stag' or 'prod'")

    # ---- Local dev only (.env optional) ----
    env_file = find_dotenv(f"{env_key}.env", usecwd=True)
    if env_file:
        load_dotenv(env_file, override=False)
        print(f"🟢 Loaded local {env_key}.env")
    else:
        print("🟡 Using injected RunPod env vars")

    if env_key == "stag":
        AWS_ACCESS_KEY_ID = os.environ["STAG_AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = os.environ["STAG_AWS_SECRET_ACCESS_KEY"]
        S3_BUCKET = os.environ["STAG_S3_BUCKET"]
    else:
        AWS_ACCESS_KEY_ID = os.environ["PROD_AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = os.environ["PROD_AWS_SECRET_ACCESS_KEY"]
        S3_BUCKET = os.environ["PROD_S3_BUCKET"]

    AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
    RUNTIME_ENV = env_key

    print(f"✅ Runtime environment configured: {env_key}")
    return RUNTIME_ENV


def classify_env(value: str, default: str = "stag") -> str:
    if not value:
        return default

    val = value.lower()
    if "prod" in val or "production" in val:
        return "prod"
    if "stag" in val or "staging" in val:
        return "stag"
    return default

