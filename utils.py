import jwt
import json


def generate_title(text: str) -> str:
    """Generate a brief title for the chat session"""
    words = text.split()
    title = ' '.join(words[:4]) if len(words) > 1 else text
    print(f"Generated title: {title}")
    return title


# Token handling functions
def decode_id_token(id_token):
    try:
        decoded_token = jwt.decode(id_token, options={"verify_signature": False})
        return decoded_token
    except jwt.ExpiredSignatureError:
        raise ValueError('id_token has expired')
    except jwt.InvalidTokenError as e:
        raise ValueError(f'Invalid id_token: {str(e)}')

def extract_user_details(decoded_token):
    email = decoded_token.get('email')
    username = decoded_token.get('name') or decoded_token.get('preferred_username') or decoded_token.get('cognito:username')
    openid = decoded_token.get('sub')

    if not email or not username:
        raise ValueError('Missing user information (email or username)')

    return {
        'id': openid,
        'email': email,
        'username': username
    }

# Helper function for structured responses
def create_response(status_code, message, headers=None):
    headers = headers or {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, id_token, Authorization',
    }
    return {
        'statusCode': status_code,
        'headers': headers,
        'body': json.dumps(message)
    }