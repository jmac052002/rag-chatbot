# Simulates what API Gateway or Lex would send to your Lambda
import orchestrator_lambda

fake_event = {
    "question": "How does FAISS perform similarity search on embeddings?"
}

# Call the handler exactly like AWS would
result = orchestrator_lambda.lambda_handler(fake_event, None)

print("Status Code:", result["statusCode"])
print("Response Body:")
import json
print(json.dumps(json.loads(result["body"]), indent=2)) 