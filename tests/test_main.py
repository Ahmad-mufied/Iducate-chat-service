import pytest
import json
from main import lambda_handler

@pytest.fixture
def valid_event():
    return {
        "headers": {
            "Content-Type": "application/json",
            "id_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItaWQiLCJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20iLCJuYW1lIjoidGVzdC11c2VyIn0.sample-signature"
        },
        "body": json.dumps({
            "prompt": "Hello, how are you?",
            "chat_id": "test-chat-123"
        })
    }

def test_lambda_handler_missing_token(valid_event):
    event = valid_event.copy()
    event["headers"] = {}
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "id_token header is missing" in response["body"]

def test_lambda_handler_missing_prompt(valid_event):
    event = valid_event.copy()
    event["body"] = json.dumps({})
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "Field 'prompt' is required" in response["body"] 