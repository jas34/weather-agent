import os
import requests
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
# Weather Tool Implementation using Weatherstack API
# --------------------------

def get_weather(city: str) -> str:
    """
    Fetches weather data for the given city using the Weatherstack API.
    """
    # Use the WEATHERSTACK_API_KEY environment variable if available,
    # otherwise default to the provided access key.
    access_key = os.getenv("WEATHERSTACK_API_KEY", "8dde6b30e3d7b6203743565676b704a9")
    base_url = "http://api.weatherstack.com/current"
    params = {
        "access_key": access_key,
        "query": city,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Check for API error responses
        if "error" in data:
            return f"Error retrieving weather data: {data['error'].get('info', 'Unknown error')}"

        # Extract weather details from the response
        location = data.get("location", {})
        current = data.get("current", {})

        city_name = location.get("name", "Unknown")
        country = location.get("country", "Unknown")
        temperature = current.get("temperature", "N/A")
        feelslike = current.get("feelslike", "N/A")
        humidity = current.get("humidity", "N/A")
        weather_descriptions = current.get("weather_descriptions", [])
        description = weather_descriptions[0] if weather_descriptions else "N/A"
        wind_speed = current.get("wind_speed", "N/A")

        # Build a formatted weather report
        weather_report = (
            f"Weather in {city_name}, {country}:\n"
            f"  Temperature: {temperature}°C (feels like {feelslike}°C)\n"
            f"  Description: {description}\n"
            f"  Humidity: {humidity}%\n"
            f"  Wind Speed: {wind_speed} km/h"
        )
        return weather_report
    except Exception as e:
        return f"Error retrieving weather data: {e}"

# Create the LangChain Tool for weather using the updated get_weather function
weather_tool = Tool(
    name="WeatherTool",
    func=get_weather,
    description="Useful for retrieving the current weather for a given city using Weatherstack API. Input should be the city name."
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
      {"content": "Weather in New York, United States: ...", "role": "assistant"}
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
