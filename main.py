import json
import os
import uuid
import logging
import datetime
import boto3
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Query, Depends
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, Field
from enum import Enum
from google.generativeai import GenerativeModel
import google.generativeai as genai
from mangum import Mangum
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
import logging

from utils import get_id_token, get_user_id
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Gemini Chatbot API",
    description="Chatbot API using Google Gemini with DynamoDB",
    version="0.0.1"
)

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["id_token", "*"],  # Explicitly add 'id_token'
)


# Load specific environment file
env_file = os.getenv('ENV', '.env')
load_dotenv(env_file)

# Example:
# ENV=.env.production python main.py

# Environment variables for configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-pro')

# AWS credentials (replace with your own access key and secret key)
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', '')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
region_name = 'ap-southeast-1'

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up DynamoDB client
dynamodb = boto3.resource(
    'dynamodb',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
table_name = os.environ.get('TABLE_NAME')  # Fetch DynamoDB table name from environment variable
table = dynamodb.Table(table_name)
print(f"DynamoDB table initialized: {table_name}")


class DetailedHeaderLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Log incoming request details
        print(f"Incoming Request: {request.method} {request.url}")
        print("Request Headers:")
        for name, value in request.headers.items():
            print(f"  {name}: {value}")

        # Process the request
        response = await call_next(request)

        # Log outgoing response details
        print(f"Response Status Code: {response.status_code}")
        print("Response Headers:")
        for name, value in response.headers.items():
            print(f"  {name}: {value}")

        return response

class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Check if the request is an OPTIONS request
        if request.method == "OPTIONS":
            # Return a response with CORS headers for OPTIONS request
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": "*",  # Adjust to your needs
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "id_token, Content-Type",  # Custom headers
                },
            )

        # If it's not an OPTIONS request, let the request pass to the next middleware/handler
        response = await call_next(request)
        # You can also add headers to non-OPTIONS requests if needed
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "id_token, Content-Type"
        return response



app.add_middleware(DetailedHeaderLoggingMiddleware)
app.add_middleware(CustomCORSMiddleware)


# Enum to represent roles in the chat
class ChatRole(str, Enum):
    """Enumerates the possible roles in a conversation (user, assistant, system)."""
    user = "user"
    assistant = "assistant"
    system = "system"


# Pydantic model for the content of a message (text)
class TextContent(BaseModel):
    """Represents the text content of a message."""
    text: str = Field(default=None, examples=["Hello!"])


# Pydantic model for a message in the chat
class Message(BaseModel):
    """Represents a message in the conversation."""
    role: ChatRole  # Role of the sender (user, assistant, etc.)
    content: list[TextContent]  # List of text content within the message


# Pydantic model for summarizing a chat session
# Pydantic model for summarizing a chat session with user_id
class ChatSummary(BaseModel):
    """Represents a summary of a chat session."""
    id: uuid.UUID = Field(default=None, examples=["12345678-1234-5678-1234-567812345678"])
    user_id: str = Field(default=None, examples=["user-123"])  # Add user_id here
    title: str = Field(default=None, examples=["Summer Destinations"])
    created_at: datetime.datetime = Field(default=None, examples=["2022-05-18T12:19:51.685496"])
    updated_at: datetime.datetime = Field(default=None, examples=["2022-05-18T12:19:51.685496"])


# Pydantic model for a full chat, including messages
class Chat(ChatSummary):
    """Represents a full chat session, including all messages exchanged."""
    messages: list[Message]  # List of messages in the chat


# Pydantic model for the response to a chat request
class ChatResponse(BaseModel):
    """Represents the response containing the latest message and chat summary."""
    message: Message
    chat: ChatSummary


# Pydantic model for chat input
class ChatInput(BaseModel):
    """
    Represents the input for initiating or continuing a chat session.
    """
    prompt: str = Field(..., example="Hello!")  # User's message
    chat_id: uuid.UUID | None = Field(
        default=None,
        description="A unique identifier for the chat session. This ID is generated when a new chat session is created and is used to retrieve and continue the chat session in future requests.",
        examples=["12345678-1234-5678-1234-567812345678"]
    )


def save_chat_to_dynamodb(chat: Chat, user_id: str):
    """Saves the provided chat object to DynamoDB, including user_id."""
    table.put_item(
        Item=json.loads(chat.model_dump_json()) | {"user_id": user_id}  # Add user_id to the chat item
    )


def append_messages_to_chat(chat_id: uuid.UUID, new_messages: list[Message]):
    """Appends new messages to the existing chat in DynamoDB in a single update."""
    try:
        # Prepare new messages to be appended
        new_messages_data = [msg.model_dump() for msg in new_messages]

        # Update the chat item by appending the new messages and updating the timestamp
        response = table.update_item(
            Key={'id': str(chat_id)},
            UpdateExpression="SET #messages = list_append(#messages, :new_messages), #updated_at = :updated_at",
            ExpressionAttributeNames={
                '#messages': 'messages',  # Attribute to update (list of messages)
                '#updated_at': 'updated_at'  # Attribute for the timestamp
            },
            ExpressionAttributeValues={
                ':new_messages': new_messages_data,  # List of new messages to append
                ':updated_at': datetime.datetime.now().isoformat()  # Update timestamp
            },
            ReturnValues="ALL_NEW"  # Return the updated item
        )
        return response.get('Attributes', None)  # Return the updated chat attributes
    except ClientError as e:
        print(f"Error updating chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to update chat")


def get_chat_from_dynamodb(chat_id: uuid.UUID) -> Chat | None:
    """Fetches a chat session from DynamoDB by its unique chat ID."""
    response = table.get_item(Key={'id': str(chat_id)})
    if 'Item' not in response:
        return None
    return Chat(**response['Item'])  # Convert response to a Chat object


def get_chats_from_dynamodb(user_id: str) -> list[Chat]:
    """Fetches all chat sessions for a specific user from DynamoDB."""
    response = table.scan(
        FilterExpression="user_id = :user_id",  # Filter by user_id
        ExpressionAttributeValues={":user_id": user_id}  # Use user_id value
    )
    return [Chat(**item) for item in response.get('Items', [])]


def generate_title(text: str) -> str:
    """Generate a brief title for the chat session"""
    words = text.split()
    title = ' '.join(words[:4]) if len(words) > 1 else text
    print(f"Generated title: {title}")
    return title


def invoke_gemini_model(messages: list[Message]) -> str:
    """Invoke Gemini model with conversation history"""
    print(f"Invoking Gemini Model: {MODEL_NAME}")
    try:
        # Prepare conversation history for Gemini
        chat_history = []
        for msg in messages[:-1]:  # Exclude the last message (current user input)
            if msg.role == 'user':
                chat_history.append({'role': 'user', 'parts': [msg.content[0].text]})
            elif msg.role == 'assistant':
                chat_history.append({'role': 'model', 'parts': [msg.content[0].text]})

        # Current user input is the last message
        current_input = messages[-1].content[0].text

        # Initialize the model
        model = GenerativeModel(MODEL_NAME)

        # Start chat with history
        chat = model.start_chat(history=chat_history)

        # Send message and get response
        response = chat.send_message(current_input)

        print(f"Gemini Response Length: {len(response.text)} characters")
        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error invoking Gemini: {str(e)}")


@app.get('/chat/{chat_id}')
async def get_chat_history(chat_id: uuid.UUID) -> list[Message]:
    """
    Retrieves the history of a specific chat session by its unique chat ID.
    """
    chat = get_chat_from_dynamodb(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Chat not found")
    return chat.messages  # Return the list of messages in the chat


@app.get('/chats')
async def get_all_chats_ids(
        id_token: str = Depends(get_id_token)  # Using the dependency to get id_token from headers
) -> list[ChatSummary]:
    """
    Retrieves a summary of all chat sessions stored in DynamoDB for a specific user.
    """
    # get sub (user_id) from token
    user_id = get_user_id(id_token)
    return get_chats_from_dynamodb(user_id)  # Return a list of ChatSummary objects filtered by user_id


@app.post('/chat', response_model=ChatResponse)
async def chat_with_the_model(
        chat_input: ChatInput,
        id_token: str = Depends(get_id_token)  # Using the dependency to get id_token from headers
) -> ChatResponse:
    """
    Initiates a chat with the model. If a chat ID is provided, continues the existing chat.
    If no chat ID is provided, a new chat session is created, and the model generates a title.
    The conversation is sent to Amazon Bedrock for processing, and the latest response is returned.
    """
    # Extract input values
    prompt = chat_input.prompt
    chat_id = chat_input.chat_id

    # get sub (user_id) from token
    user_id = get_user_id(id_token)

    if chat_id is None:
        # Generate a new chat session
        chat_id = uuid.uuid4()
        title = generate_title(prompt)

        chat = Chat(
            id=chat_id,
            user_id=user_id,
            messages=[],
            title=title,
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat()
        )

        # Save the initial chat
        save_chat_to_dynamodb(chat, user_id)
    else:
        # Retrieve existing chat session
        chat = get_chat_from_dynamodb(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

    # Append user message
    user_message = Message(role="user", content=[TextContent(text=prompt)])
    chat.messages.append(user_message)

    # Invoke Gemini Model
    try:
        gemini_response_text = invoke_gemini_model(chat.messages)
    except Exception as e:
        logger.error(f"Model invocation failed: {e}")
        raise

    # Create assistant message
    assistant_message = Message(
        role="assistant",
        content=[TextContent(text=gemini_response_text)]
    )
    chat.messages.append(assistant_message)

    # Append both user and assistant messages
    append_messages_to_chat(chat_id, [user_message, assistant_message])

    # Return only the latest assistant message and chat summary
    return ChatResponse(
        message=assistant_message,
        chat=ChatSummary(
            id=chat.id,
            user_id=chat.user_id,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at
        ),
    )


if __name__ == "__main__":
    # Run the FastAPI app using uvicorn when running locally
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # Use Mangum to handle requests for AWS Lambda (serverless deployment)
    handler = Mangum(app, lifespan="off")
