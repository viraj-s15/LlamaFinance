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
from langchain.llms import HuggingFacePipeline
from langchain.tools import Tool, tool
from pydantic import BaseModel, Field
import sys
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)

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
    huggingface_access_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=huggingface_access_token)
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")

app = FastAPI()

class UserInput(BaseModel):
    text: str

@app.post("/chat")
async def chat_endpoint(user_input: UserInput, verbose: bool = False, use_flash_attention: bool = False, temperature: float = 0.01, model: str = "TheBloke/zephyr-7B-alpha-AWQ", max_iterations: int = 3, message_history: int = 5, cache_dir: str = None, max_new_tokens: int = 512, repetition_penalty: float = 1.1):
    model = AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, use_flash_attention2=use_flash_attention)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
