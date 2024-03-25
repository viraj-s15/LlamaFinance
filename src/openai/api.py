from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.tools import Tool, tool
from pydantic import BaseModel, Field
import sys
import argparse
import logging

logging.basicConfig(level=logging.ERROR)

try:
    sys.path.append("../../src/")
    from tools.stock_business_info_tools import stock_business_tools
    from tools.stock_data_tools import stock_data_tools
except ImportError as e:
    logging.error(f"Import error: {e}")
    stock_business_tools = None
    stock_data_tools = None

try:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")

app = FastAPI()

class UserInput(BaseModel):
    text: str

@app.post("/chat")
async def chat_endpoint(user_input: UserInput, verbose: bool = False, temperature: float = 0.0, model: str = "gpt-3.5-turbo", max_iterations: int = 3, message_history: int = 5):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=temperature,
        model_name=model,
    )

    tools = stock_data_tools + stock_business_tools

    conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=message_history,
        return_messages=True
    )

    agent = initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=verbose,
        max_iterations=max_iterations,
        early_stopping_method="generate",
        memory=conversational_memory,
        handle_parsing_errors="Check your output and make sure it conforms!",
    )

    response = agent(user_input.text)
    return {"response": response["output"]}

@app.get("/test")
async def test(verbose: bool = False, temperature: float = 0.0, model: str = "gpt-3.5-turbo", max_iterations: int = 3, message_history: int = 5):
    try:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=temperature,
            model_name=model,
        )

        tools = stock_data_tools + stock_business_tools

        conversational_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=message_history,
            return_messages=True
        )

        agent = initialize_agent(
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=verbose,
            max_iterations=max_iterations,
            early_stopping_method="generate",
            memory=conversational_memory,
            handle_parsing_errors="Check your output and make sure it conforms!",
        )

        return {"message": "Initialization successful"}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
