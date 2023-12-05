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
