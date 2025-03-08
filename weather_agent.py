import os
import time
import uuid
import asyncio
import json
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

# Load environment variables from a .env file if available
load_dotenv()

# --------------------------
# Pydantic models for messages
# --------------------------

class Message(BaseModel):
    content: str
    role: str

# --------------------------
# Additional Pydantic models for Chat Completions endpoint (non-streaming)
# --------------------------

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[Message]
    temperature: Optional[float] = 0.7  # Optional temperature setting

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# --------------------------
# Weather Tool Implementation (Mock Response)
# --------------------------

def get_weather(city: str) -> str:
    """
    Mocks weather data for the given city.
    """
    return (
        f"Mock weather in {city}:\n"
        "  Temperature: 1°C (feels like -5°C)\n"
        "  Description: Clear skies\n"
        "  Humidity: 40%\n"
        "  Wind Speed: 10 km/h"
    )

# Create the LangChain Tool for weather using the mock get_weather function
weather_tool = Tool(
    name="WeatherTool",
    func=get_weather,
    description="Useful for retrieving the current weather for a given city (mock implementation). Input should be the city name."
)

# --------------------------
# LangChain Agent Initialization
# --------------------------

# Ensure the OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize the OpenAI language model
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Create the LangChain agent with the weather tool
agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# --------------------------
# FastAPI Application Setup
# --------------------------

app = FastAPI()

@app.post("/chat", response_model=List[Message])
async def chat(messages: List[Message]) -> List[Message]:
    """
    Chat endpoint that accepts a list of messages and returns the assistant's response.
    """
    try:
        # Use the last user message as the prompt for the agent
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            return [Message(content="No user message provided.", role="assistant")]
        prompt = user_messages[-1].content

        # Run the LangChain agent
        agent_response = await agent.arun(prompt)
        return [Message(content=agent_response, role="assistant")]
    except Exception as e:
        return [Message(content=f"Error processing the request: {e}", role="assistant")]

@app.get("/health")
async def health():
    """
    Health check endpoint that returns a JSON indicating the service is running.
    """
    return {"status": "healthy"}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Non-streaming endpoint compatible with OpenAI's Chat Completions API.
    Expects a request body with a list of messages and returns a response in a similar format.
    """
    try:
        # Use the last user message as the prompt for the agent
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            error_message = Message(content="No user message provided.", role="assistant")
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(index=0, message=error_message, finish_reason="error")],
                usage=ChatCompletionUsage()
            )
            return response

        prompt = user_messages[-1].content

        # Run the LangChain agent (the agent doesn't currently use temperature from the request)
        agent_response = await agent.arun(prompt)

        assistant_message = Message(content=agent_response, role="assistant")
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=assistant_message,
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=500,        # Dummy token count
                completion_tokens=400,    # Dummy token count
                total_tokens=900          # Dummy token count
            )
        )
        return response
    except Exception as e:
        error_message = Message(content=f"Error processing the request: {e}", role="assistant")
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionChoice(index=0, message=error_message, finish_reason="error")],
            usage=ChatCompletionUsage()
        )

async def stream_response_chunks(response_text: str, response_id: str, model: str) -> AsyncGenerator[str, None]:
    """
    Async generator that simulates streaming by yielding chunks of a final response.
    Each chunk is formatted as a JSON object similar to OpenAI's streaming format.
    """
    timestamp = int(time.time())
    system_fingerprint = "fp_44709d6fcb"

    # Initial chunk with the role information and empty content
    init_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None
            }
        ]
    }
    yield json.dumps(init_chunk) + "\n"
    await asyncio.sleep(0.1)

    # Simulate streaming by splitting the response into words.
    words = response_text.split()
    for word in words:
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "system_fingerprint": system_fingerprint,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word},
                    "logprobs": None,
                    "finish_reason": None
                }
            ]
        }
        yield json.dumps(chunk) + "\n"
        await asyncio.sleep(0.1)

    # Final chunk indicating completion.
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "logprobs": None,
                "finish_reason": "stop"
            }
        ]
    }
    yield json.dumps(final_chunk) + "\n"

@app.post("/chat/completions")
async def stream_chat_completions(request: ChatCompletionRequest):
    """
    Streaming endpoint compliant with OpenAI's /chat/completions API.
    It streams newline-delimited JSON chunks simulating token-by-token delivery.
    """
    # Extract the last user message as the prompt.
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "No user message provided."},
                    "logprobs": None,
                    "finish_reason": "error"
                }
            ]
        }
        return StreamingResponse(iter([json.dumps(error_chunk) + "\n"]), media_type="application/json")

    prompt = user_messages[-1].content

    try:
        # Run the LangChain agent to get the full response.
        agent_response = await agent.arun(prompt)
    except Exception as e:
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"Error processing the request: {e}"},
                    "logprobs": None,
                    "finish_reason": "error"
                }
            ]
        }
        return StreamingResponse(iter([json.dumps(error_chunk) + "\n"]), media_type="application/json")

    response_id = f"chatcmpl-{uuid.uuid4()}"
    return StreamingResponse(
        stream_response_chunks(agent_response, response_id, request.model),
        media_type="application/json"
    )

if __name__ == "__main__":
    uvicorn.run("weather_agent:app", host="0.0.0.0", port=8080, reload=True)
