# Managing Imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel, Field
import requests
import pandas as pd
import yfinance as yf

# Relative imports from the tools directory
from tools.stock_data_tools import stock_data_tools
from tools.stock_business_info_tools import stock_business_tools

# setup import from env variables
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--temperature", type=float, default=0.0)
# set temperature to a variable

args = parser.parse_args()

temperature = 0.0
if args.temperature:
    temperature = args.temperature

verbose = False
if args.verbose:
    verbose = True


llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    # temperature is set to 0 to prevent the model from generating a response that is not related to the stock market
    temperature=temperature,
    # gpt-3.5-turbo because that is the only api key I have
    model_name="gpt-3.5-turbo",
)

tools = stock_data_tools + stock_business_tools

# This is where we store the conversation history
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

agent = initialize_agent(
    # This is the only only agent which supports multi-input structured tools
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=verbose,
    # We only let the model rethink its response 3 times
    max_iterations=3,
    early_stopping_method="generate",
    memory=conversational_memory,
)

while True:
    user_input = input(">>> ")
    if user_input == "exit":
        break
    response = agent(user_input)
    print(response["output"])
