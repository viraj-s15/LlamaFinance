# Managing Imports
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

logging.basicConfig(level=logging.INFO)

try:
    sys.path.append("../../src/")
    from tools.stock_business_info_tools import stock_business_tools
    from tools.stock_data_tools import stock_data_tools
except ImportError as e:
    logging.error(f"Import error: {e}")
    stock_business_tools = None
    stock_data_tools = None

# setup import from env variables
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")


parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
parser.add_argument("--max_iterations", type=int, default=3)
parser.add_argument("--message_history", type=str, default=5)

args = parser.parse_args()

verbose = False
if args.verbose:
    verbose = True


llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    # temperature is set to 0 to prevent the model from generating a response that is not related to the stock market
    temperature=args.temperature,
    # gpt-3.5-turbo because that is the only api key I have
    model_name=args.model,
)

tools = stock_data_tools + stock_business_tools

# This is where we store the conversation history
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=args.message_history, return_messages=True
)

agent = initialize_agent(
    # This is the only only agent which supports multi-input structured tools
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=verbose,
    max_iterations=args.max_iterations,
    early_stopping_method="generate",
    memory=conversational_memory,
    handle_parsing_errors="Check your output and make sure it conforms!",
)

while True:
    user_input = input(">>> ")
    if user_input == "exit":
        break
    response = agent(user_input)
    print(response["output"])
