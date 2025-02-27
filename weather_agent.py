import os
#from requests import get  # No longer needed for mocking
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file if available
load_dotenv()

# --------------------------
# Pydantic models for messages
# --------------------------

class Message(BaseModel):
    content: str
    role: str

# --------------------------
# Weather Tool Implementation (Mock Response)
# --------------------------

def get_weather(city: str) -> str:
    """
    Mocks weather data for the given city.
    """
    # Return a hardcoded weather report for testing purposes.
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
    agent="zero-shot-react-description",  # Agent type that uses tool descriptions
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

    Request Body Example:
    [
      {"content": "What's the weather like in New York?", "role": "user"}
    ]

    Response Body Example:
    [
      {"content": "Mock weather in New York:\n  Temperature: 25°C (feels like 25°C)\n  ...", "role": "assistant"}
    ]
    """
    try:
        # Use the last user message as the prompt for the agent
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            return [Message(content="No user message provided.", role="assistant")]
        prompt = user_messages[-1].content

        # Run the LangChain agent
        agent_response = agent.run(prompt)
        return [Message(content=agent_response, role="assistant")]
    except Exception as e:
        return [Message(content=f"Error processing the request: {e}", role="assistant")]

@app.get("/health")
async def health():
    """
    Health check endpoint that returns a JSON indicating the service is running.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("weather_agent:app", host="0.0.0.0", port=8080, reload=True)
