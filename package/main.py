import json
import os

import boto3
import uuid
import datetime
from botocore.exceptions import BotoCoreError, ClientError
import logging

from utils import decode_id_token, extract_user_details, create_response, generate_title

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("test-chat")  # Replace with your DynamoDB table name

# Fixed agent details
AGENT_ID = os.getenv('AGENT_ID', "")
AGENT_ALIAS_ID = os.getenv('AGENT_ALIAS_ID', "")


def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    status_code = 200
    headers = {"Content-Type": "application/json"}

    # Extract id_token
    id_token_header = event['headers'].get('id_token')
    if not id_token_header:
        return create_response(400, {'error': 'id_token header is missing'})

    try:
        decoded_token = decode_id_token(id_token_header)
        user_details = extract_user_details(decoded_token)
        user_id = user_details['id']
    except ValueError as e:
        return create_response(400, {'error': str(e)})
    except Exception as e:
        return create_response(500, {'error': 'Internal server error', 'message': str(e)})

    try:
        # Parse and validate input
        if "body" not in event:
            raise ValueError("Request body is missing.")

        body = json.loads(event["body"])
        user_prompt = body.get("prompt", "")
        chat_id = body.get("chat_id", str(uuid.uuid4()))

        if not user_prompt:
            raise ValueError("Field 'prompt' is required.")

        # Initialize Bedrock client
        try:
            bedrock_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        except (BotoCoreError, ClientError) as e:
            print(f"Error initializing Bedrock client: {e}")
            raise Exception("Failed to initialize Bedrock client.")

        # Generate a session ID
        session_id = str(uuid.uuid4())

        response = bedrock_client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=user_prompt,
        )

        event_stream = response['completion']
        final_answer = None
        try:
            for event in event_stream:
                if 'chunk' in event:
                    data = event['chunk']['bytes']
                    final_answer = data.decode('utf8')
                    logger.info(f"Final answer ->\n{final_answer}")
                    end_event_received = True
                elif 'trace' in event:
                    logger.info(json.dumps(event['trace'], indent=2))
                else:
                    raise Exception("unexpected event.", event)
        except Exception as e:
            raise Exception("unexpected event.", e)
        # Create chat object
        chat_response = {
            "id": chat_id,
            "user_id": user_id,
            "title": generate_title(user_prompt),
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": final_answer}
            ]
        }

        # Save chat to DynamoDB
        try:
            table.put_item(Item={
                "id": chat_id,
                "user_id": user_id,
                "created_at": chat_response["created_at"],
                "updated_at": chat_response["updated_at"],
                "title": chat_response["title"],
                "messages": chat_response["messages"]
            })
            print(f"Chat saved to DynamoDB: {chat_response}")
        except (BotoCoreError, ClientError) as e:
            print(f"Error saving chat to DynamoDB: {e}")
            raise Exception("Error saving chat to DynamoDB.")

        # Return the response
        response_body = {
            "response": final_answer,
            "chat": chat_response
        }

    except ValueError as ve:
        print(f"Validation error: {ve}")
        status_code = 400
        response_body = {"error": str(ve)}
    except Exception as ex:
        print(f"Error: {ex}")
        status_code = 500
        response_body = {"error": str(ex)}

    # Ensure proper serialization of response
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": json.dumps(response_body)  # Changed from {"response": response} to response_body
    }
