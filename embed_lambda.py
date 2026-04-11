import os
import json
import boto3
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== CONFIG =====
REGION = "us-east-1"
BUCKET_NAME = "agentic-docs-repo-joseph"
INDEX_PREFIX = "indexes/"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024

# ===== /tmp PATHS =====
# Lambda writes here during execution
# Persists across warm invocations
FAISS_LOCAL = "/tmp/faiss_index.bin"
METADATA_LOCAL = "/tmp/metadata.json"

# ===== AWS CLIENTS =====
# Outside handler for warm start performance
s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# ===== TEXT SPLITTER =====
# Same settings as your original script
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)


# ===== HELPER FUNCTIONS =====
def get_embedding(text):
    """Convert text chunk to 1024-dimensional vector via Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["embedding"]


def load_existing_index():
    """
    Download existing FAISS index and metadata from S3.
    If they don't exist yet this is the first run —
    create a fresh empty index instead.
    """
    try:
        s3.download_file(
            BUCKET_NAME,
            f"{INDEX_PREFIX}faiss_index.bin",
            FAISS_LOCAL
        )
        s3.download_file(
            BUCKET_NAME,
            f"{INDEX_PREFIX}metadata.json",
            METADATA_LOCAL
        )
        index = faiss.read_index(FAISS_LOCAL)
        with open(METADATA_LOCAL, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded existing index with {index.ntotal} vectors")
        return index, metadata

    except Exception:
        # First time running — no index exists yet
        print("No existing index found — creating fresh index")
        return faiss.IndexFlatL2(EMBEDDING_DIM), []


def save_and_upload_index(index, metadata):
    """Save updated index locally then push both files back to S3."""
    faiss.write_index(index, FAISS_LOCAL)
    with open(METADATA_LOCAL, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    s3.upload_file(FAISS_LOCAL, BUCKET_NAME, f"{INDEX_PREFIX}faiss_index.bin")
    s3.upload_file(METADATA_LOCAL, BUCKET_NAME, f"{INDEX_PREFIX}metadata.json")
    print(f"Index uploaded to S3 — total vectors: {index.ntotal}")


# ===== LAMBDA HANDLER =====
def lambda_handler(event, context):
    """
    Triggered by S3 when a new .txt file is uploaded.

    S3 event payload looks like:
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "agentic-docs-repo-joseph"},
                "object": {"key": "what_is_rag.txt"}
            }
        }]
    }
    """
    try:
        processed = []
        skipped = []

        # Step 1 — extract file info from S3 event
        # S3 can batch multiple records in one event
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = record["s3"]["object"]["key"]

            # Only process .txt files
            if not key.endswith(".txt"):
                print(f"Skipping non-txt file: {key}")
                skipped.append(key)
                continue

            print(f"Processing: {key}")

            # Step 2 — download the new txt file from S3
            response = s3.get_object(Bucket=bucket, Key=key)
            text = response["Body"].read().decode("utf-8")

            # Step 3 — chunk it
            # Same splitter settings as your original script
            chunks = splitter.split_text(text)
            print(f"Split into {len(chunks)} chunks")

            # Step 4 — load existing index so we ADD to it
            # not overwrite everything from scratch
            index, metadata = load_existing_index()

            # Step 5 — embed each chunk and add to index
            added = 0
            for chunk in chunks:
                try:
                    vector = get_embedding(chunk)
                    index.add(np.array([vector], dtype="float32"))
                    metadata.append({
                        "text": chunk,
                        "source": key
                    })
                    added += 1
                except Exception as e:
                    print(f"Skipped chunk due to error: {e}")

            # Step 6 — save and push back to S3
            save_and_upload_index(index, metadata)
            processed.append({"file": key, "chunks_added": added})

        return {
            "statusCode": 200,
            "body": json.dumps({
                "processed": processed,
                "skipped": skipped
            })
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        } 