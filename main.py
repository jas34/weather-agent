import weather_agent

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("weather_agent:app", host="0.0.0.0", port=8080, reload=True)
