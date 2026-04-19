import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import botocore
except ImportError:
    botocore = None

try:
    import faiss
except ImportError:
    faiss = None


# ===== CONFIG =====
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
DYNAMO_TABLE_NAME = os.getenv("DYNAMO_TABLE_NAME", "agent_interactions")

REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is not set")

LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID", "anthropic.claude-haiku-4-5-20251001-v1:0")
USE_INFERENCE_PROFILE = os.getenv("BEDROCK_USE_INFERENCE_PROFILE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
SONNET_FALLBACK_MODEL_ID = os.getenv(
    "BEDROCK_SONNET_FALLBACK_MODEL_ID", "anthropic.claude-sonnet-4-20250514-v1:0"
)
NOVA_LITE_FALLBACK_MODEL_ID = os.getenv(
    "BEDROCK_NOVA_LITE_MODEL_ID", "amazon.nova-lite-v1:0"
)
NOVA_MICRO_FALLBACK_MODEL_ID = os.getenv(
    "BEDROCK_NOVA_MICRO_MODEL_ID", "amazon.nova-micro-v1:0"
)

MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "500"))
TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("BEDROCK_TOP_P", "0.9"))

# === Local / S3 FAISS configuration ===
S3_BUCKET_NAME = os.getenv("FAISS_S3_BUCKET")
S3_FAISS_KEY = "indexes/faiss_index.bin"
S3_METADATA_KEY = "indexes/metadata.json"
LOCAL_FAISS_PATH = "faiss_index.bin"
LOCAL_METADATA_PATH = "metadata.json"


# ===== Initialize =====
index = None
metadata: List[Dict] = []
bedrock = None
dynamodb = None
print(f"{Fore.CYAN}Initializing environment...{Style.RESET_ALL}")


# ===== AWS clients =====
def initialize_aws_clients() -> None:
    """Initialize AWS clients when boto3 is available."""
    global bedrock, dynamodb

    if not boto3:
        print(f"{Fore.YELLOW}boto3 not installed, running mock mode.{Style.RESET_ALL}")
        bedrock = None
        dynamodb = None
        return

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        dynamodb = boto3.client("dynamodb", region_name=REGION)
        print(f"{Fore.GREEN}AWS clients initialized successfully.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}AWS initialization error: {exc}{Style.RESET_ALL}")
        bedrock = None
        dynamodb = None


# ===== FAISS file management =====
def load_files_from_s3() -> bool:
    """Download FAISS and metadata from S3 bucket."""
    if not boto3:
        print(f"{Fore.RED}boto3 not installed, cannot fetch from S3.{Style.RESET_ALL}")
        return False

    try:
        s3 = boto3.client("s3", region_name=REGION)
        print(f"{Fore.CYAN}Downloading FAISS and metadata from '{S3_BUCKET_NAME}'...{Style.RESET_ALL}")
        s3.download_file(S3_BUCKET_NAME, S3_FAISS_KEY, LOCAL_FAISS_PATH)
        s3.download_file(S3_BUCKET_NAME, S3_METADATA_KEY, LOCAL_METADATA_PATH)
        print(f"{Fore.GREEN}Files downloaded from S3 successfully.{Style.RESET_ALL}")
        return True
    except Exception as exc:
        if botocore and isinstance(exc, botocore.exceptions.ClientError):
            err_code = exc.response.get("Error", {}).get("Code", "Unknown")
            if err_code == "NoSuchKey":
                print(f"{Fore.RED}Missing FAISS or metadata file in S3.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}S3 download failed: {exc}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}S3 retrieval error: {exc}{Style.RESET_ALL}")
        return False


def upload_to_s3() -> None:
    """Upload local FAISS and metadata to S3."""
    if not boto3:
        print(f"{Fore.RED}boto3 not installed, cannot upload to S3.{Style.RESET_ALL}")
        return

    missing_files = [
        path for path in (LOCAL_FAISS_PATH, LOCAL_METADATA_PATH) if not os.path.exists(path)
    ]
    if missing_files:
        print(
            f"{Fore.RED}Upload cancelled. Missing local file(s): "
            f"{', '.join(missing_files)}{Style.RESET_ALL}"
        )
        return

    try:
        s3 = boto3.client("s3", region_name=REGION)
        s3.upload_file(LOCAL_FAISS_PATH, S3_BUCKET_NAME, S3_FAISS_KEY)
        s3.upload_file(LOCAL_METADATA_PATH, S3_BUCKET_NAME, S3_METADATA_KEY)
        print(f"{Fore.GREEN}Uploaded FAISS & metadata to '{S3_BUCKET_NAME}'.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}Upload to S3 failed: {exc}{Style.RESET_ALL}")


# ===== Vector store loading =====
def choose_data_source() -> str:
    print(f"{Fore.CYAN}Choose data source for FAISS & metadata:{Style.RESET_ALL}")
    print("1) Local files")
    print("2) S3 bucket")
    return input("Enter 1 or 2 [default=1]: ").strip() or "1"


def load_vector_assets(choice: str) -> None:
    global index, metadata

    if choice == "2":
        if not load_files_from_s3():
            print(f"{Fore.YELLOW}Falling back to local files...{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}Using local FAISS & metadata files.{Style.RESET_ALL}")

    index = None
    metadata = []

    try:
        if faiss and os.path.exists(LOCAL_FAISS_PATH):
            index = faiss.read_index(LOCAL_FAISS_PATH)
            print(f"{Fore.GREEN}FAISS index loaded successfully.{Style.RESET_ALL}")
        elif not faiss:
            print(f"{Fore.YELLOW}faiss not installed; vector search disabled.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}FAISS index missing; mock retrieval active.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}Error loading FAISS: {exc}{Style.RESET_ALL}")
        index = None

    try:
        if os.path.exists(LOCAL_METADATA_PATH):
            with open(LOCAL_METADATA_PATH, "r", encoding="utf-8") as file_obj:
                loaded = json.load(file_obj)
            if isinstance(loaded, list):
                metadata = loaded
                print(f"{Fore.GREEN}Loaded metadata.json ({len(metadata)} records).{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}metadata.json is not a list; ignoring it.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}metadata.json not found locally.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}Error loading metadata: {exc}{Style.RESET_ALL}")
        metadata = []


# ===== Bedrock health check =====
def check_bedrock() -> None:
    if not boto3 or not bedrock:
        print(f"{Fore.YELLOW}Skipping Bedrock health check (AWS client unavailable).{Style.RESET_ALL}")
        return

    try:
        control = boto3.client("bedrock", region_name=REGION)
        models = control.list_foundation_models().get("modelSummaries", [])
        ids = {model["modelId"] for model in models}

        if EMBED_MODEL_ID in ids:
            print(f"{Fore.GREEN}Embedding model '{EMBED_MODEL_ID}' is listed in Bedrock.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Embedding model '{EMBED_MODEL_ID}' not listed; verify access.{Style.RESET_ALL}")

        if USE_INFERENCE_PROFILE:
            print(
                f"{Fore.CYAN}LLM is configured to use an inference profile ARN. "
                f"That resource will be validated on first invocation.{Style.RESET_ALL}"
            )
        elif LLM_MODEL_ID in ids:
            print(f"{Fore.GREEN}LLM model '{LLM_MODEL_ID}' is listed in Bedrock.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Model '{LLM_MODEL_ID}' not listed; verify access.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}Bedrock check failed: {exc}{Style.RESET_ALL}")


# ===== DynamoDB table ensure =====
def ensure_table() -> None:
    if not dynamodb:
        return

    try:
        names = dynamodb.list_tables().get("TableNames", [])
        if DYNAMO_TABLE_NAME in names:
            return

        print(f"{Fore.YELLOW}Creating DynamoDB table '{DYNAMO_TABLE_NAME}'...{Style.RESET_ALL}")
        dynamodb.create_table(
            TableName=DYNAMO_TABLE_NAME,
            AttributeDefinitions=[
                {"AttributeName": "userId", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "userId", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        waiter = dynamodb.get_waiter("table_exists")
        waiter.wait(TableName=DYNAMO_TABLE_NAME)
        print(f"{Fore.GREEN}Table is ready.{Style.RESET_ALL}")
    except Exception as exc:
        if "ResourceInUseException" not in str(exc):
            print(f"{Fore.YELLOW}DynamoDB table check failed: {exc}{Style.RESET_ALL}")


# ===== Embedding / retrieval =====
def mock_embedding_dim() -> int:
    if index is not None and hasattr(index, "d"):
        return int(index.d)
    return 1536


def embed_text(text: str) -> np.ndarray:
    if not bedrock:
        return np.random.rand(mock_embedding_dim()).astype("float32")

    try:
        body = json.dumps({"inputText": text})
        response = bedrock.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return np.array(payload["embedding"], dtype="float32")
    except Exception as exc:
        print(f"{Fore.YELLOW}Embedding failed, using mock vector: {exc}{Style.RESET_ALL}")
        return np.random.rand(mock_embedding_dim()).astype("float32")


def retrieve_chunks(query: str, k: int = 3) -> List[str]:
    if index is None or not metadata:
        return [f"[Mock Doc] Relevant info for '{query}'. (no FAISS index)"]

    try:
        vec = embed_text(query)
        _, indices = index.search(np.array([vec]), k)
        results = []
        for idx in indices[0]:
            if idx < len(metadata):
                item = metadata[idx]
                if isinstance(item, dict):
                    results.append(item.get("text", str(item)))
                else:
                    results.append(str(item))
        return results or [f"[No matching metadata text for '{query}']"]
    except Exception as exc:
        print(f"{Fore.YELLOW}Retrieval failed, using mock result: {exc}{Style.RESET_ALL}")
        return [f"[Mock Doc] Relevant info for '{query}'. (retrieval fallback)"]


# ===== Prompt builder =====
def construct_prompt(user_input: str, chunks: List[str]) -> str:
    prompt = f"User question: {user_input}\n\nRelevant documents:\n"
    for chunk in chunks:
        prompt += f"- {chunk}\n"
    return prompt.strip()


# ===== Query Bedrock LLM =====
def model_supports_claude_45_param_restriction(model_id: str) -> bool:
    lower_model_id = model_id.lower()
    return "claude-haiku-4-5" in lower_model_id or "claude-sonnet-4-5" in lower_model_id


def build_inference_config(model_id: str) -> Dict:
    config = {
        "maxTokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if not model_supports_claude_45_param_restriction(model_id):
        config["topP"] = TOP_P
    return config


def invoke_converse_model(model_id: str, prompt: str) -> str:
    response = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=build_inference_config(model_id),
    )

    output = response.get("output", {}).get("message", {}).get("content", [])
    text_parts = [block.get("text", "") for block in output if "text" in block]
    return "".join(text_parts).strip() or "[No response returned]"


def invoke_text_model(model_id: str, prompt: str) -> str:
    body = json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "topP": TOP_P,
            },
        }
    )
    response = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    return payload.get("results", [{}])[0].get("outputText", "[No output returned]")


def call_model(model_id: str, prompt: str) -> str:
    lower_model_id = model_id.lower()
    if lower_model_id.startswith("arn:aws:bedrock:") or "claude" in lower_model_id or "nova" in lower_model_id:
        return invoke_converse_model(model_id, prompt)
    return invoke_text_model(model_id, prompt)


def build_candidate_models() -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    if USE_INFERENCE_PROFILE:
        candidates.append((INFERENCE_PROFILE_ARN, "primary inference profile"))
        candidates.append((NOVA_LITE_FALLBACK_MODEL_ID, "Nova Lite fallback"))
        candidates.append((NOVA_MICRO_FALLBACK_MODEL_ID, "Nova Micro fallback"))
    else:
        candidates.append((LLM_MODEL_ID, "primary model"))
        candidates.append((SONNET_FALLBACK_MODEL_ID, "Claude Sonnet fallback"))
        candidates.append((NOVA_LITE_FALLBACK_MODEL_ID, "Nova Lite fallback"))
        candidates.append((NOVA_MICRO_FALLBACK_MODEL_ID, "Nova Micro fallback"))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for model_id, label in candidates:
        if model_id and model_id not in seen:
            deduped.append((model_id, label))
            seen.add(model_id)
    return deduped


def query_llm(prompt: str) -> str:
    if not bedrock:
        return f"[MOCK RESPONSE] Simulated output for: {prompt[:50]}..."

    last_error = None
    for model_id, label in build_candidate_models():
        try:
            if label != "primary inference profile" and label != "primary model":
                print(f"{Fore.YELLOW}Trying {label}: {model_id}{Style.RESET_ALL}")
            return call_model(model_id, prompt)
        except Exception as exc:
            last_error = exc
            print(f"{Fore.RED}Bedrock API error using {label}: {exc}{Style.RESET_ALL}")
            time.sleep(0.25)

    return f"[Error contacting Bedrock: {last_error}]"


# ===== DynamoDB helpers =====
def save_interaction(user_id: str, question: str, response_text: str) -> None:
    if not dynamodb:
        return

    try:
        dynamodb.put_item(
            TableName=DYNAMO_TABLE_NAME,
            Item={
                "userId": {"S": user_id},
                "timestamp": {"S": datetime.utcnow().isoformat()},
                "question": {"S": question},
                "response": {"S": response_text},
            },
        )
    except Exception as exc:
        print(f"{Fore.RED}[ERROR] Save interaction failed: {exc}{Style.RESET_ALL}")


def query_user_history_items(user_id: str) -> List[Dict]:
    if not dynamodb:
        return []

    items: List[Dict] = []
    exclusive_start_key = None

    while True:
        params = {
            "TableName": DYNAMO_TABLE_NAME,
            "KeyConditionExpression": "userId = :u",
            "ExpressionAttributeValues": {":u": {"S": user_id}},
            "ScanIndexForward": False,
        }
        if exclusive_start_key:
            params["ExclusiveStartKey"] = exclusive_start_key

        response = dynamodb.query(**params)
        items.extend(response.get("Items", []))
        exclusive_start_key = response.get("LastEvaluatedKey")
        if not exclusive_start_key:
            break

    return items


def fetch_user_history(user_id: str, limit: Optional[int] = None) -> None:
    if not dynamodb:
        print("DynamoDB not configured.")
        return

    try:
        items = query_user_history_items(user_id)
        if not items:
            print("No history found.")
            return

        if limit is not None:
            items = items[:limit]

        for item in items:
            print(
                f"\n{item['timestamp']['S']}"
                f"\n{item.get('question', {}).get('S', '[No question]')}"
                f"\n{item.get('response', {}).get('S', '[No response]')}"
            )
    except Exception as exc:
        print(f"{Fore.RED}[ERROR] Fetch history failed: {exc}{Style.RESET_ALL}")


def chunked(sequence: List[Dict], size: int) -> List[List[Dict]]:
    return [sequence[i : i + size] for i in range(0, len(sequence), size)]


def clear_user_history(user_id: str) -> None:
    if not dynamodb:
        print("DynamoDB not configured.")
        return

    try:
        items = query_user_history_items(user_id)
        if not items:
            print("No history to clear.")
            return

        delete_requests = [
            {
                "DeleteRequest": {
                    "Key": {
                        "userId": item["userId"],
                        "timestamp": item["timestamp"],
                    }
                }
            }
            for item in items
        ]

        for batch in chunked(delete_requests, 25):
            request_items = {DYNAMO_TABLE_NAME: batch}
            response = dynamodb.batch_write_item(RequestItems=request_items)
            unprocessed = response.get("UnprocessedItems", {})
            retries = 0
            while unprocessed and retries < 5:
                time.sleep(0.5 * (2 ** retries))
                response = dynamodb.batch_write_item(RequestItems=unprocessed)
                unprocessed = response.get("UnprocessedItems", {})
                retries += 1

        print(f"{Fore.GREEN}History cleared for {user_id}.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}[ERROR] Clear history failed: {exc}{Style.RESET_ALL}")


def list_all_users() -> List[str]:
    if not dynamodb:
        print("DynamoDB not configured.")
        return []

    try:
        users = set()
        scan_kwargs = {
            "TableName": DYNAMO_TABLE_NAME,
            "ProjectionExpression": "userId",
        }
        exclusive_start_key = None

        while True:
            if exclusive_start_key:
                scan_kwargs["ExclusiveStartKey"] = exclusive_start_key
            elif "ExclusiveStartKey" in scan_kwargs:
                del scan_kwargs["ExclusiveStartKey"]

            response = dynamodb.scan(**scan_kwargs)
            for item in response.get("Items", []):
                value = item.get("userId", {}).get("S")
                if value:
                    users.add(value)

            exclusive_start_key = response.get("LastEvaluatedKey")
            if not exclusive_start_key:
                break

        sorted_users = sorted(users)
        if not sorted_users:
            print("No users found.")
            return []

        print(f"{Fore.CYAN}👥 Users:{Style.RESET_ALL}")
        for user in sorted_users:
            print(f"  - {user}")
        return sorted_users
    except Exception as exc:
        print(f"{Fore.RED}[ERROR] List users failed: {exc}{Style.RESET_ALL}")
        return []


def count_user_questions(user_id: str) -> int:
    if not dynamodb:
        print("DynamoDB not configured.")
        return 0

    try:
        items = query_user_history_items(user_id)
        count = len(items)
        print(f"{Fore.CYAN}Total questions for {user_id}: {count}{Style.RESET_ALL}")
        return count
    except Exception as exc:
        print(f"{Fore.RED}[ERROR] Count questions failed: {exc}{Style.RESET_ALL}")
        return 0


def delete_user_and_records(user_id: str) -> None:
    if not dynamodb:
        print("DynamoDB not configured.")
        return

    try:
        items = query_user_history_items(user_id)
        if not items:
            print(f"📭 No records found for {user_id}.")
            return

        delete_requests = [
            {
                "DeleteRequest": {
                    "Key": {
                        "userId": item["userId"],
                        "timestamp": item["timestamp"],
                    }
                }
            }
            for item in items
        ]

        for batch in chunked(delete_requests, 25):
            request_items = {DYNAMO_TABLE_NAME: batch}
            response = dynamodb.batch_write_item(RequestItems=request_items)
            unprocessed = response.get("UnprocessedItems", {})
            retries = 0
            while unprocessed and retries < 5:
                time.sleep(0.5 * (2 ** retries))
                response = dynamodb.batch_write_item(RequestItems=unprocessed)
                unprocessed = response.get("UnprocessedItems", {})
                retries += 1

        print(f"{Fore.GREEN}Deleted user '{user_id}' and all associated records.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}[ERROR] Delete user failed: {exc}{Style.RESET_ALL}")


# ===== MAIN LOOP =====
def print_help() -> None:
    print(f"{Fore.CYAN}Available commands:{Style.RESET_ALL}")
    print("  switch <user>          - change active user ID")
    print("  history [n]            - show saved history, optionally limited to n items")
    print("  clear history          - delete saved history for current user")
    print("  list users             - list all users found in DynamoDB")
    print("  delete user <user>     - delete a user and all associated records")
    print("  count questions <user> - count total saved questions for a user")
    print("  upload faiss           - upload local FAISS + metadata files to S3")
    print("  help                   - show this message")
    print("  exit                   - quit the program")


def main() -> None:
    initialize_aws_clients()
    load_vector_assets(choose_data_source())
    check_bedrock()
    ensure_table()

    user_id = input(f"{Fore.CYAN}Enter user ID: {Style.RESET_ALL}").strip()
    if not user_id:
        user_id = "default-user"
        print(f"{Fore.YELLOW}Blank user ID detected. Using '{user_id}'.{Style.RESET_ALL}")

    print(
        f"{Fore.CYAN}Chat started. Type 'help' for commands, "
        f"or 'exit' to quit.{Style.RESET_ALL}"
    )

    while True:
        user_input = input(f"\n{Fore.YELLOW}👤 ({user_id}) You:{Style.RESET_ALL} ").strip()
        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered in ["exit", "quit"]:
            print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break

        if lowered == "help":
            print_help()
            continue

        if lowered.startswith("switch "):
            new_user = user_input.split(" ", 1)[1].strip()
            if new_user:
                user_id = new_user
                print(f"{Fore.CYAN}Switched to '{user_id}'.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Please provide a user ID after 'switch'.{Style.RESET_ALL}")
            continue

        if lowered.startswith("history"):
            parts = user_input.split()
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            fetch_user_history(user_id, limit)
            continue

        if lowered == "clear history":
            confirm = input(f"Delete all history for {user_id}? (yes/no): ").strip().lower()
            if confirm == "yes":
                clear_user_history(user_id)
            else:
                print("Clear history cancelled.")
            continue

        if lowered == "list users":
            list_all_users()
            continue

        if lowered.startswith("delete user "):
            target_user = user_input.split(" ", 2)[2].strip()
            if target_user:
                confirm = input(f"Delete user '{target_user}' and all records? (yes/no): ").strip().lower()
                if confirm == "yes":
                    delete_user_and_records(target_user)
                else:
                    print("Delete user cancelled.")
            else:
                print(f"{Fore.YELLOW}Please provide a user ID after 'delete user'.{Style.RESET_ALL}")
            continue

        if lowered.startswith("count questions "):
            target_user = user_input.split(" ", 2)[2].strip()
            if target_user:
                count_user_questions(target_user)
            else:
                print(f"{Fore.YELLOW}Please provide a user ID after 'count questions'.{Style.RESET_ALL}")
            continue

        if lowered == "upload faiss":
            upload_to_s3()
            continue

        chunks = retrieve_chunks(user_input)
        prompt = construct_prompt(user_input, chunks)
        llm_response = query_llm(prompt)
        print(f"\n{Fore.GREEN}LLM Response:{Style.RESET_ALL}\n{llm_response}")
        save_interaction(user_id, user_input, llm_response)
        print(f"{Fore.GREEN}Response saved.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
