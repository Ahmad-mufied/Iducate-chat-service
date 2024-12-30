# AWS Lambda JWT Chat Function

A serverless AWS Lambda function that handles chat interactions using JWT authentication and DynamoDB for storage.

## Overview

This Lambda function provides a secure chat API endpoint that:
- Validates JWT tokens for authentication
- Processes chat messages using AWS Bedrock
- Stores chat history in DynamoDB
- Handles error cases and provides appropriate responses

## Prerequisites

- Python 3.9+
- AWS Account with access to:
  - Lambda
  - DynamoDB
  - Bedrock
- Required Python packages:
  - boto3
  - PyJWT
  - pydantic

## Installation

1. Create a new directory for your Lambda package:
```bash
mkdir package
```

2. Install required packages to the package directory:
```bash
pip install --target ./package boto3 PyJWT pydantic
```

3. Copy Python files to the package directory:
```bash
cp -r ./*.py ./package
```

4. Create deployment package:
```bash
zip -r9 lambda_function.zip ./package
```

## Configuration

### Environment Variables

The following environment variables need to be set in your Lambda function:

- `AGENT_ID`: Your AWS Bedrock Agent ID
- `AGENT_ALIAS_ID`: Your AWS Bedrock Agent Alias ID

Example `.env` file:
```env
# AWS Bedrock Configuration
AGENT_ID=your_agent_id_here
AGENT_ALIAS_ID=your_agent_alias_id_here

# DynamoDB Configuration
DYNAMODB_TABLE=test-chat
AWS_REGION=us-east-1

# JWT Configuration (optional)
JWT_SECRET_KEY=your_secret_key_here
JWT_ALGORITHM=HS256

# AWS Credentials (for local development)
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

> **Note**: Never commit your actual `.env` file to version control. Add `.env` to your `.gitignore` file.

### DynamoDB Table

Create a DynamoDB table with the following structure:
- Table Name: `test-chat` (configurable)
- Partition Key: `id` (String)
- Sort Key: `user_id` (String)

## API Reference

This Lambda function exposes a single endpoint for chat interactions.

### Available Endpoints

#### POST /chat
Primary endpoint for chat interactions with the AI assistant.

**Purpose:**
- Process user messages
- Generate AI responses using AWS Bedrock
- Store chat history in DynamoDB
- Manage chat sessions

#### Authentication
JWT token-based authentication is required for all requests.

#### Headers
| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Content-Type` | string | Yes | Must be `application/json` |
| `id_token` | string | Yes | Valid JWT authentication token |

#### Request Body
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | The user's message or question |
| `chat_id` | string | No | Existing chat ID for continuing a conversation |

Example Request:
```json
{
  "prompt": "Your message here",
  "chat_id": "optional-existing-chat-id"
}
```

#### Response Structure
| Field | Type | Description |
|-------|------|-------------|
| `response` | string | The AI assistant's response |
| `chat` | object | Chat session details |
| `chat.id` | string | Unique chat identifier |
| `chat.user_id` | string | User identifier from JWT |
| `chat.title` | string | Auto-generated chat title |
| `chat.created_at` | string | Creation timestamp (ISO) |
| `chat.updated_at` | string | Last update timestamp (ISO) |
| `chat.messages` | array | Message history |

Example Success Response:
```json
{
  "response": "AI response message",
  "chat": {
    "id": "chat-uuid",
    "user_id": "user-id",
    "title": "Chat title",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z",
    "messages": [
      {
        "role": "user",
        "content": "user message"
      },
      {
        "role": "assistant", 
        "content": "AI response"
      }
    ]
  }
}
```

#### Error Responses
| Status Code | Description | Example Response |
|-------------|-------------|------------------|
| 400 | Bad Request | `{"error": "Field 'prompt' is required"}` |
| 401 | Unauthorized | `{"error": "Invalid id_token: token has expired"}` |
| 500 | Server Error | `{"error": "Internal server error", "message": "..."}` |

#### Implementation Notes
- All requests require JWT authentication
- Chat history is automatically persisted to DynamoDB
- Chat titles are generated from the first message
- Responses are streamed from AWS Bedrock

## Error Handling

The API returns appropriate HTTP status codes:

- 200: Success
- 400: Bad Request (invalid input)
- 401: Unauthorized (invalid token)
- 500: Internal Server Error

## Development

## Setup with SAM

### Step 1: Initialize Project Structure

1. Create project directory and set up virtual environment:
```bash
# Create project directory
mkdir aws-lambda-chat
cd aws-lambda-chat

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install required packages:
```bash
pip install boto3 PyJWT pydantic aws-sam-cli
pip freeze > requirements.txt
```

### Step 2: Set Up Lambda Code Structure

1. Create the package directory and copy your code:
```bash
# Create directories
mkdir package
mkdir events

# Copy Lambda function files
cp main.py package/
cp utils.py package/
```

2. Create SAM template (template.yaml):
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  ChatFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./package
      Handler: main.lambda_handler
      Runtime: python3.9
      Timeout: 30
      MemorySize: 256
      Environment:
        Variables:
          AGENT_ID: your_agent_id_here
          AGENT_ALIAS_ID: your_agent_alias_id_here
          DYNAMODB_TABLE: test-chat
      Policies:
        - DynamoDBCrudPolicy:
            TableName: test-chat
        - Statement:
            - Effect: Allow
              Action:
                - bedrock:*
              Resource: '*'
      Events:
        ChatAPI:
          Type: Api
          Properties:
            Path: /chat
            Method: post
```

### Step 3: Configure Test Events

1. Create a test event file (events/event.json):
```json
{
  "body": "{\"prompt\": \"Hello, how are you?\"}",
  "headers": {
    "Content-Type": "application/json",
    "id_token": "your.test.token"
  }
}
```

2. Create environment file (env.json):
```json
{
  "ChatFunction": {
    "AGENT_ID": "your_agent_id_here",
    "AGENT_ALIAS_ID": "your_agent_alias_id_here",
    "DYNAMODB_TABLE": "test-chat",
    "AWS_REGION": "us-east-1"
  }
}
```

### Step 4: Local Development Setup

1. Start local DynamoDB:
```bash
docker run -p 8000:8000 amazon/dynamodb-local
```

2. Create test table:
```bash
aws dynamodb create-table \
    --table-name test-chat \
    --attribute-definitions \
        AttributeName=id,AttributeType=S \
        AttributeName=user_id,AttributeType=S \
    --key-schema \
        AttributeName=id,KeyType=HASH \
        AttributeName=user_id,KeyType=RANGE \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
    --endpoint-url http://localhost:8000
```

### Step 5: Build and Test

1. Build the SAM application:
```bash
sam build
```

2. Test locally:
```bash
# Test single invocation
sam local invoke ChatFunction \
    -e events/event.json \
    -n env.json

# Start local API
sam local start-api \
    -n env.json \
    --warm-containers EAGER
```

3. Test the API endpoint:
```bash
curl -X POST http://localhost:3000/chat \
    -H "Content-Type: application/json" \
    -H "id_token: your.test.token" \
    -d '{"prompt": "Hello, how are you?"}'
```

### Step 6: Deploy to AWS

1. First-time deployment:
```bash
sam deploy --guided
```

2. Follow the prompts:
   - Stack Name: `aws-lambda-chat`
   - AWS Region: your preferred region
   - Confirm changes before deploy: Yes
   - Allow SAM CLI IAM role creation: Yes
   - Save arguments to configuration file: Yes

3. Subsequent deployments:
```bash
sam deploy
```

### Project Structure
```
aws-lambda-chat/
├── package/
│   ├── main.py              # Lambda handler
│   └── utils.py             # Utility functions
├── events/
│   └── event.json           # Test event
├── template.yaml            # SAM template
├── main.py              # Lambda handler
│── utils.py             # Utility functions
├── env.json                 # Local environment variables
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

### Important Notes

1. **Local Testing**:
   - Ensure Docker is running for SAM local testing
   - Use local DynamoDB for development
   - JWT tokens can be mocked for testing

2. **Environment Variables**:
   - Update env.json with your actual values
   - Never commit sensitive values to version control

3. **AWS Credentials**:
   - Configure AWS credentials using `aws configure`
   - Ensure proper IAM permissions for deployment

## Security

- All requests must include a valid JWT token
- Tokens are validated for:
  - Expiration
  - Required claims (email, username)
  - Signature (if configured)

## License

MIT License - See LICENSE file for details
