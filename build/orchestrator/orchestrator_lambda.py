import os
import json
import re
import boto3
import numpy as np
import faiss

# ===== CONFIG =====
REGION = "us-east-1"
BUCKET_NAME = "agentic-docs-repo-joseph"
INDEX_PREFIX = "indexes/"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ===== LAMBDA /tmp PATHS =====
# Lambda has no persistent filesystem
# /tmp is the only writable directory
# It persists between warm invocations (same container reuse)
FAISS_LOCAL = "/tmp/faiss_index.bin"
METADATA_LOCAL = "/tmp/metadata.json"

# ===== AWS CLIENTS =====
# These are initialized OUTSIDE lambda_handler
# This is intentional — on warm starts the container
# is reused and these clients don't need to be recreated
# This is a Lambda performance best practice
s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# ===== INDEX LOADER =====
# Also outside lambda_handler for the same reason
# On cold start this downloads from S3
# On warm start the files already exist in /tmp
def load_index():
    """Download FAISS index and metadata from S3 if not already in /tmp."""
    if not os.path.exists(FAISS_LOCAL):
        print("Cold start — downloading faiss_index.bin from S3...")
        s3.download_file(BUCKET_NAME, f"{INDEX_PREFIX}faiss_index.bin", FAISS_LOCAL)

    if not os.path.exists(METADATA_LOCAL):
        print("Cold start — downloading metadata.json from S3...")
        s3.download_file(BUCKET_NAME, f"{INDEX_PREFIX}metadata.json", METADATA_LOCAL)

    index = faiss.read_index(FAISS_LOCAL)
    with open(METADATA_LOCAL, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

# Load once at container init — not inside handler
# This means warm invocations skip the S3 download entirely
index, metadata = load_index()


# ===== CORE FUNCTIONS =====
# These are identical to your script — nothing changes here
def embed_text(text):
    """Embed query using Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    return np.array(json.loads(response["body"].read())["embedding"], dtype="float32")


def retrieve_chunks(query, k=3):
    """Search FAISS for top k matching chunks."""
    vec = embed_text(query)
    _, I = index.search(np.array([vec]), k)
    return [metadata[i]["text"] for i in I[0] if i < len(metadata)]


def construct_prompt(user_input, chunks):
    """Build the structured prompt with retrieved context."""
    prompt = f"""
Human: Answer the following question based on the provided documents.
If the answer is not in the documents, say you don't know.

User question: {user_input}

Relevant documents:
"""
    for i, chunk in enumerate(chunks, 1):
        prompt += f"- Doc{i}: {chunk}\n"
    prompt += "\nAvailable tools: NotifyHR, SummarizeDoc, CreateTask\nAssistant:"
    return prompt.strip()


def query_llm(prompt):
    """Call Claude via Bedrock."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 500,
        "temperature": 0.7
    })
    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    data = json.loads(response["body"].read())
    if "content" in data:
        if isinstance(data["content"], list):
            return data["content"][0].get("text", "[No text found]")
        return str(data["content"])
    return "[No response]"


def detect_action(response_text):
    """Check if Claude wants to trigger a tool."""
    m = re.search(r"\[ACTION: (\w+)\]", response_text)
    return m.group(1) if m else None


# ===== LAMBDA HANDLER =====
# This replaces main() entirely
# event = the JSON payload sent by Lex or API Gateway
# context = Lambda runtime info (timeout remaining, function name, etc)
def lambda_handler(event, context):
    """
    Expected event payload:
    {
        "question": "How does FAISS work?"
    }
    """
    try:
        # Step 1 — get the question from the event
        # Previously this was: user_input = input("You: ")
        # Now it comes from the JSON payload
        user_input = event.get("question", "").strip()

        if not user_input:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No question provided in event payload"})
            }

        # Steps 2-5 are identical to your script
        chunks = retrieve_chunks(user_input)
        prompt = construct_prompt(user_input, chunks)
        llm_response = query_llm(prompt)
        action = detect_action(llm_response)

        # Step 6 — return result as JSON
        # Previously this was: print(llm_response)
        # Lambda must return a dict with statusCode and body
        return {
            "statusCode": 200,
            "body": json.dumps({
                "question": user_input,
                "answer": llm_response,
                "action_detected": action,
                "chunks_retrieved": len(chunks)
            })
        }

    except Exception as e:
        # Always handle exceptions in Lambda
        # Unhandled exceptions return a 500 with no useful info
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }