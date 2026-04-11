import embed_lambda

# Simulates what S3 sends when a file is uploaded
fake_event = {
    "Records": [{
        "s3": {
            "bucket": {"name": "agentic-docs-repo-joseph"},
            "object": {"key": "what_is_rag.txt"}
        }
    }]
}

result = embed_lambda.lambda_handler(fake_event, None)

print("Status Code:", result["statusCode"])
import json
print(json.dumps(json.loads(result["body"]), indent=2)) 