# RAG Chatbot AWS Bedrock + FAISS

A production-style Retrieval-Augmented Generation (RAG) pipeline built on AWS. Documents stored in S3 are chunked, vectorized using Amazon Titan Embeddings V2, and indexed with FAISS for semantic search. Queries are answered by Claude via Amazon Bedrock using only retrieved context — no hallucination from stale training data.

Built as part of a cloud/AI engineering bootcamp (Digital Cloud Training), with a focus on real AWS infrastructure, not toy examples.

---

## Architecture

```
S3 (source docs)
     │
     ▼
Titan Embeddings V2      ← converts text chunks to 1024-d vectors
     │
     ▼
FAISS Index              ← similarity search (L2 distance)
     │
     ▼
Retrieved chunks
     │
     ▼
Claude (Amazon Bedrock)  ← generates grounded answer from context
```

| Service | Role | Status |
|---|---|---|
| Amazon S3 | Stores `.txt` docs + FAISS index | Done |
| Titan Embeddings V2 | Converts chunks → 1024-d vectors | Done |
| FAISS | Similarity search over vectors | Done |
| AWS Lambda | Orchestrator + embedding trigger | In progress |
| Amazon Lex | NLP / intent parsing (front door) | Coming |
| Amazon Bedrock (Claude) | Generates answers from context | Coming |
| DynamoDB | Conversation memory | Future |

---

## Project Structure

```
rag-chatbot/
├── embed_documents_v3_s3.py     # Step 1: chunk docs, build FAISS index, push to S3
├── embed_documents_v5_s3.py     # Step 1 (v5): adds PDF support via PyMuPDF
├── query_retriever_v2_s3.py     # Step 2: semantic search against FAISS index
├── embed_lambda.py              # Lambda handler: triggers embedding on S3 upload
├── orchestrator_lambda.py       # Lambda handler: routes queries, calls Bedrock
├── agent_v5.py                  # Agent with tool use (v5)
├── agent_v6.py                  # Agent with tool use + DynamoDB memory (v6)
├── test_embed_local.py          # Local test harness for embed_lambda
├── test_lambda_local.py         # Local test harness for orchestrator_lambda
├── build/                       # Lambda deployment package (gitignored)
├── package/                     # Lambda dependency layer (gitignored)
├── .gitignore
└── README.md

# Generated locally (gitignored — rebuilt from S3):
# faiss_index.bin
# metadata.json
```

---

## Prerequisites

- Python 3.11 (via deadsnakes PPA on Ubuntu 24)
- AWS CLI configured with a named profile
- Amazon Bedrock model access enabled in your region:
  - Titan Embeddings V2
  - Claude (via cross-region inference profile)

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/jmac052002/rag-chatbot.git
cd rag-chatbot
```

**2. Create and activate a virtual environment**

```bash
python3.11 -m venv ~/ai-agent-env
source ~/ai-agent-env/bin/activate
```

**3. Install dependencies**

```bash
pip install boto3 faiss-cpu numpy langchain-text-splitters colorama
```

**4. Set environment variables**

```bash
export AWS_PROFILE=default
export BEDROCK_REGION=us-east-1
export BUCKET_NAME=your-s3-bucket-name
export BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
export BEDROCK_INFERENCE_PROFILE_ARN=arn:aws:bedrock:us-east-1::foundation-model/...
```

Or copy `.env.example` (if provided) to `.env` and populate it.

---

## Usage

**Every session activate the venv first:**

```bash
source ~/ai-agent-env/bin/activate
```

**Step 1 Build the FAISS index from your S3 documents:**

```bash
python embed_documents_v3_s3.py
```

This reads `.txt` files from S3, chunks them (700 tokens, 100 overlap), generates embeddings via Titan Embeddings V2, and writes `faiss_index.bin` + `metadata.json` back to `s3://your-bucket/indexes/`.

**Step 2 Test semantic retrieval:**

```bash
python query_retriever_v2_s3.py
```

Runs sample queries against the FAISS index and prints the top-k most relevant chunks with similarity scores.

**Upload new documents to S3:**

```bash
aws s3 cp ./docs/ s3://your-bucket-name/ --recursive --exclude "*" --include "*.txt"
```

---

## Key Concepts

| Term | What it means |
|---|---|
| Chunk | 700-token piece of a document. Each gets its own embedding. |
| Overlap | 100 tokens shared between adjacent chunks so context isn't lost at boundaries. |
| Embedding | 1024 floating-point numbers representing the meaning of a chunk. |
| FAISS index | A searchable map of all chunk vectors held in memory. |
| L2 score | Euclidean distance between vectors. Lower = more similar = better match. |
| k=3 | Return the 3 nearest chunks per query. |
| RAG | Retrieve chunks → inject into prompt → grounded answer from the LLM. |

---

## Lambda Deployment Note

Dependencies must be built for Lambda's runtime (Amazon Linux 2, x86_64). Do **not** zip your local venv the compiled binaries will not load on Lambda.

Build the package correctly:

```bash
pip install \
  --platform manylinux2014_x86_64 \
  --target ./package \
  --only-binary=:all: \
  numpy faiss-cpu boto3
```

Then copy your handler into `./package/` and zip the whole directory.

---

## .gitignore

The following are excluded and must be rebuilt locally or pulled from S3:

```
ai-agent-env/
faiss_index.bin
metadata.json
build/
package/
__pycache__/
*.pyc
.env
```

---

## Author

Joseph McCoy transitioning from manufacturing into cloud/DevOps engineering.  
Building real AWS infrastructure as part of the Digital Cloud Training Agentic AI bootcamp.

GitHub: [@jmac052002](https://github.com/jmac052002)
