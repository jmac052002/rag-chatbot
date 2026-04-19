import boto3
import json
import faiss
import numpy as np
import os

# ──────────────────────────────
# Configuration
# ──────────────────────────────
REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is not set")

INDEX_PREFIX = "indexes/"  # where embed_documents.py uploaded results
MODEL_ID = "amazon.titan-embed-text-v2:0"

FAISS_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.json"

# ──────────────────────────────
# AWS Clients
# ──────────────────────────────
s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# ──────────────────────────────
# File Management
# ──────────────────────────────
def download_from_s3(filename):
    """Download FAISS or metadata file from S3 if not present or outdated."""
    s3_key = f"{INDEX_PREFIX}{filename}"
    try:
        print(f"Downloading {s3_key} from S3...")
        s3.download_file(BUCKET_NAME, s3_key, filename)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {s3_key}: {e}")
        raise SystemExit(f"Missing required file: {filename}")

def ensure_local_files():
    """Ensure FAISS index and metadata exist locally."""
    for file_name in [FAISS_FILE, METADATA_FILE]:
        if not os.path.exists(file_name):
            download_from_s3(file_name)
        else:
            print(f"Using local cached {file_name}")

# ──────────────────────────────
# Embedding + Search Logic
# ──────────────────────────────
def embed_query(text: str):
    """Generate embedding for a query using Bedrock Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype="float32")

def search_index(index, query_vec, k=3):
    """Perform vector similarity search using FAISS."""
    D, I = index.search(np.array([query_vec]), k)
    return I[0], D[0]

# ──────────────────────────────
# Main Interactive Loop
# ──────────────────────────────
def main():
    ensure_local_files()

    print("Loading FAISS index and metadata...")
    index = faiss.read_index(FAISS_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Ask your agent a question (or type 'exit'):")

    while True:
        user_input = input("\n You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("Retrieving relevant documents...")
        try:
            query_vec = embed_query(user_input)
            top_indexes, scores = search_index(index, query_vec, k=3)

            for rank, (idx, score) in enumerate(zip(top_indexes, scores), 1):
                if idx < len(metadata):
                    print(f"\n Result #{rank} (score: {score:.2f}):")
                    print(f"Source: {metadata[idx]['source']}")
                    print(f"Chunk: {metadata[idx]['text'][:300]}...")  # trimmed
                else:
                    print(f"\n Invalid index {idx}")

        except Exception as e:
            print(f"Query failed: {e}")

if __name__ == "__main__":
    main()