import boto3
import json
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# ──────────────────────────────
# Configuration
# ──────────────────────────────
REGION = "us-east-1"
BUCKET_NAME = "agentic-docs-repo-joseph"
PREFIX = ""  # optional folder containing input text files
OUTPUT_PREFIX = "indexes/"  # 🔥 all output files go here
EMBEDDING_DIM = 1024  # Titan Embeddings output size

FAISS_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.json"

# ──────────────────────────────
# AWS Clients
# ──────────────────────────────
s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# ──────────────────────────────
# Text Splitter
# ──────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

# ──────────────────────────────
# Helper Functions
# ──────────────────────────────
def get_embedding(text: str):
    """Generate embedding vector using Amazon Titan via Bedrock."""
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    response_body = response["body"].read()
    return json.loads(response_body)["embedding"]

def process_file_from_s3(key: str):
    """Fetch a text file from S3 and split into chunks."""
    response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    text = response["Body"].read().decode("utf-8")
    return splitter.split_text(text)

def upload_to_s3(local_file: str, s3_key: str):
    """Upload a local file to S3."""
    try:
        s3.upload_file(local_file, BUCKET_NAME, s3_key)
        print(f"☁️ Uploaded {local_file} → s3://{BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"⚠️ Failed to upload {local_file}: {e}")

# ──────────────────────────────
# Main
# ──────────────────────────────
def main():
    print("📦 Listing S3 files...")
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    files = [obj["Key"] for obj in objects.get("Contents", []) if obj["Key"].endswith(".txt")]

    if not files:
        print("⚠️ No .txt files found in S3 bucket path.")
        return

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []

    for file_key in files:
        print(f"🔍 Processing: {file_key}")
        chunks = process_file_from_s3(file_key)

        for chunk in chunks:
            try:
                vector = get_embedding(chunk)
                index.add(np.array([vector], dtype="float32"))
                metadata.append({"text": chunk, "source": file_key})
            except Exception as e:
                print(f"⚠️ Skipped a chunk due to error: {e}")

    print("💾 Saving FAISS index and metadata locally...")
    faiss.write_index(index, FAISS_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Upload to S3 under OUTPUT_PREFIX
    print("☁️ Uploading results to S3...")
    upload_to_s3(FAISS_FILE, f"{OUTPUT_PREFIX}{FAISS_FILE}")
    upload_to_s3(METADATA_FILE, f"{OUTPUT_PREFIX}{METADATA_FILE}")

    print("✅ Done! Files saved locally and uploaded to S3.")
    print(f"   → s3://{BUCKET_NAME}/{OUTPUT_PREFIX}{FAISS_FILE}")
    print(f"   → s3://{BUCKET_NAME}/{OUTPUT_PREFIX}{METADATA_FILE}")

if __name__ == "__main__":
    main()