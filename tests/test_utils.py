import pytest
from utils import generate_title, decode_id_token, extract_user_details

def test_generate_title():
    # Test with short text
    assert generate_title("Hello world") == "Hello world"
    
    # Test with long text
    long_text = "This is a very long message that should be truncated"
    assert generate_title(long_text) == "This is a very"
    
    # Test with single word
    assert generate_title("Hello") == "Hello"

def test_decode_id_token():
    # Test valid token
    valid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItaWQiLCJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20iLCJuYW1lIjoidGVzdC11c2VyIn0.sample-signature"
    decoded = decode_id_token(valid_token)
    assert decoded["sub"] == "test-user-id"
    
    # Test invalid token
    with pytest.raises(ValueError):
        decode_id_token("invalid-token")

def test_extract_user_details():
    # Test valid token data
    token_data = {
        "sub": "test-user-id",
        "email": "test@example.com",
        "name": "Test User"
    }
    user_details = extract_user_details(token_data)
    assert user_details["id"] == "test-user-id"
    assert user_details["email"] == "test@example.com"
    assert user_details["username"] == "Test User"
    
    # Test missing required fields
    with pytest.raises(ValueError):
        extract_user_details({}) 