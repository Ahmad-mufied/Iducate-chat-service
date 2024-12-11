import jwt
from fastapi import Request, HTTPException


def get_id_token(request: Request) -> str:
    """Extracts the `id_token` from the headers."""
    id_token = request.headers.get("id_token")  # Extract the id_token from headers
    if not id_token:
        raise HTTPException(status_code=401, detail="Missing id_token in headers")
    return id_token

def get_user_id(id_token: str) -> str:
    """Extracts the `sub` claim from the `id_token`."""
    decoded_token = decode_id_token(id_token)
    return decoded_token.get("sub")

# Token handling functions
def decode_id_token(id_token) -> dict:
    try:
        decoded_token = jwt.decode(id_token, options={"verify_signature": False})
        return decoded_token
    except jwt.ExpiredSignatureError:
        raise ValueError('id_token has expired')
    except jwt.InvalidTokenError as e:
        raise ValueError(f'Invalid id_token: {str(e)}')
