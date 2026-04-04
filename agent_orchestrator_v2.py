import os
import json
import re
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

try:
    import faiss
    import boto3
except ImportError:
    faiss = None
    boto3 = None

# ===== CONFIG =====
REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ===== LOAD FAISS INDEX & METADATA =====
index = None
metadata = []

print(f"{Fore.CYAN}🔍 Initializing environment...{Style.RESET_ALL}")

try:
    if faiss and os.path.exists("faiss_index.bin"):
        index = faiss.read_index("faiss_index.bin")
        print(f"{Fore.GREEN}✅ FAISS index loaded successfully.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️ FAISS not available or index missing.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}❌ Error loading FAISS index: {e}{Style.RESET_ALL}")

try:
    if os.path.exists("metadata.json"):
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"{Fore.GREEN}✅ Loaded metadata.json ({len(metadata)} records).{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️ metadata.json not found.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}❌ Error loading metadata.json: {e}{Style.RESET_ALL}")

# ===== BEDROCK CLIENT =====
if boto3:
    try:
        bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        print(f"{Fore.GREEN}✅ Bedrock client initialized successfully.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Unable to initialize Bedrock client: {e}{Style.RESET_ALL}")
        bedrock = None
else:
    bedrock = None
    print(f"{Fore.YELLOW}⚠️ boto3 not installed.{Style.RESET_ALL}")


# ===== CORE FUNCTIONS =====
def embed_text(text):
    if not bedrock:
        return np.random.rand(1024).astype("float32")
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    return np.array(json.loads(response["body"].read())["embedding"], dtype="float32")


def retrieve_chunks(query, k=3):
    if not index or not metadata:
        return [f"[Mock Doc] Relevant info for '{query}' (no FAISS index)."]
    vec = embed_text(query)
    _, I = index.search(np.array([vec]), k)
    return [metadata[i]["text"] for i in I[0] if i < len(metadata)]


def construct_prompt(user_input, chunks):
    prompt = f"""
Human: Answer the following question based on the provided documents.

User question: {user_input}

Relevant documents:
"""
    for chunk in chunks:
        prompt += f"- {chunk}\n"
    prompt += "\nAvailable tools: NotifyHR, SummarizeDoc, CreateTask\nAssistant:"
    return prompt.strip()


def query_llm(prompt):
    if not bedrock:
        return f"[MOCK RESPONSE] Simulated answer for: {prompt[:60]}..."

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 500,
        "temperature": 0.7
    })

    try:
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
        return json.dumps(data, indent=2)

    except Exception as e:
        print(f"{Fore.RED}❌ Bedrock API error: {e}{Style.RESET_ALL}")
        return f"[Error contacting Bedrock: {e}]"


def detect_action(response_text):
    m = re.search(r"\[ACTION: (\w+)\]", response_text)
    return m.group(1) if m else None


def main():
    print(f"{Fore.CYAN}🤖 C3PO at your service! Type a question (or 'exit' to quit).{Style.RESET_ALL}")
    while True:
        user_input = input(f"\n{Fore.YELLOW}🧠 You:{Style.RESET_ALL} ")
        if user_input.lower() in ["exit", "quit"]:
            print(f"{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
            break

        chunks = retrieve_chunks(user_input)
        prompt = construct_prompt(user_input, chunks)
        llm_response = query_llm(prompt)

        print(f"\n{Fore.GREEN}📨 LLM Response:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{llm_response}{Style.RESET_ALL}")

        action = detect_action(llm_response)
        if action:
            print(f"\n{Fore.BLUE}⚙️ Detected action: {action} (simulated Lambda call){Style.RESET_ALL}")
        else:
            print(f"\n{Fore.GREEN}✅ No tool action detected.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()