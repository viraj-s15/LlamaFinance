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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
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
    huggingface_access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    login(token=huggingface_access_token)
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")

app = FastAPI()

class UserInput(BaseModel):
    text: str

<<<<<<< HEAD
@app.get("/test")
async def test(verbose: bool = False, temperature: float = 0.0, model: str = "mistralai/Mistral-7B-Instruct-v0.1", max_iterations: int = 3, message_history: int = 5, cache_dir: str = None, max_new_tokens: int = 512, repetition_penalty: float = 1.1, load_in_4bit: bool = False, load_in_8bit: bool = False, quant_dtype_4bit: str = "fp4", compute_dtype_4bit: str = "fp16", device: str = "0"):
    logging.info("Starting test endpoint")
    try:
        logging.info("Checking compute dtype and load in 8bit")
        if compute_dtype_4bit and load_in_8bit:
            raise Exception("8bit quant only supports int8, other dtypes can only be used with 4bit quant")

        logging.info("Setting compute dtype for 4bit quant")
        compute_dtype_4bit = None
        if compute_dtype_4bit == "fp16":
            compute_dtype_4bit = torch.float16
        elif compute_dtype_4bit == "fp32":
            compute_dtype_4bit = torch.float32
        elif compute_dtype_4bit == "fp64":
            compute_dtype_4bit = torch.float64
        else:
            raise Exception("Invalid compute dtype for 4bit quant")

        logging.info("Checking quant dtype and load in 8bit")
        if quant_dtype_4bit and load_in_8bit:
            raise Exception("8bit quant only supports int8, fp4 can only be used with 4bit quant")

        logging.info("Checking load in 4bit and 8bit")
        if load_in_4bit and load_in_8bit:
            raise Exception("Cannot load in both 4bit and 8bit")

        logging.info("Setting BitsAndBytesConfig")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_dtype_4bit,
            bnb_4bit_compute_dtype=torch.float16,
        )

        logging.info("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        logging.info("Creating text generation pipeline")
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            device=device,
        )

        logging.info("Creating HuggingFacePipeline")
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        logging.info("Combining tools")
        tools = stock_data_tools + stock_business_tools

        logging.info("Creating conversational memory")
        conversational_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=message_history,
            return_messages=True
        )

        logging.info("Initializing agent")
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

        logging.info("Initialization successful")
        return {"message": "Initialization successful"}

    except Exception as e:
        logging.error(f"Error in test endpoint: {e}")
        return {"error": str(e)}

=======
>>>>>>> 7300f5b (feat: Add logging to test endpoint)
@app.get("/test")
async def test(verbose: bool = False, temperature: float = 0.0, model: str = "mistralai/Mistral-7B-Instruct-v0.1", max_iterations: int = 3, message_history: int = 5, cache_dir: str = None, max_new_tokens: int = 512, repetition_penalty: float = 1.1, load_in_4bit: bool = False, load_in_8bit: bool = False, quant_dtype_4bit: str = "fp4", compute_dtype_4bit: str = "fp16", device: str = "0"):
    logging.info("Starting test endpoint")
    try:
        logging.info("Checking compute dtype and load in 8bit")
        if compute_dtype_4bit and load_in_8bit:
            raise Exception("8bit quant only supports int8, other dtypes can only be used with 4bit quant")

        logging.info("Setting compute dtype for 4bit quant")
        compute_dtype_4bit = None
        if compute_dtype_4bit == "fp16":
            compute_dtype_4bit = torch.float16
        elif compute_dtype_4bit == "fp32":
            compute_dtype_4bit = torch.float32
        elif compute_dtype_4bit == "fp64":
            compute_dtype_4bit = torch.float64
        else:
            raise Exception("Invalid compute dtype for 4bit quant")

        logging.info("Checking quant dtype and load in 8bit")
        if quant_dtype_4bit and load_in_8bit:
            raise Exception("8bit quant only supports int8, fp4 can only be used with 4bit quant")

        logging.info("Checking load in 4bit and 8bit")
        if load_in_4bit and load_in_8bit:
            raise Exception("Cannot load in both 4bit and 8bit")

        logging.info("Setting BitsAndBytesConfig")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_dtype_4bit,
            bnb_4bit_compute_dtype=torch.float16,
        )

        logging.info("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        logging.info("Creating text generation pipeline")
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            device=device,
        )

        logging.info("Creating HuggingFacePipeline")
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        logging.info("Combining tools")
        tools = stock_data_tools + stock_business_tools

        logging.info("Creating conversational memory")
        conversational_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=message_history,
            return_messages=True
        )

        logging.info("Initializing agent")
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

        logging.info("Initialization successful")
        return {"message": "Initialization successful"}

    except Exception as e:
        logging.error(f"Error in test endpoint: {e}")
        return {"error": str(e)}

import logging

@app.get("/test")
async def test(verbose: bool = False, temperature: float = 0.0, model: str = "mistralai/Mistral-7B-Instruct-v0.1", max_iterations: int = 3, message_history: int = 5, cache_dir: str = None, max_new_tokens: int = 512, repetition_penalty: float = 1.1, load_in_4bit: bool = False, load_in_8bit: bool = False, quant_dtype_4bit: str = "fp4", compute_dtype_4bit: str = "fp16", device: str = "0"):
    logging.info("Starting test endpoint")
    try:
        logging.info("Checking compute dtype and load in 8bit")
        if compute_dtype_4bit and load_in_8bit:
            raise Exception("8bit quant only supports int8, other dtypes can only be used with 4bit quant")

        logging.info("Setting compute dtype for 4bit quant")
        compute_dtype_4bit = None
        if compute_dtype_4bit == "fp16":
            compute_dtype_4bit = torch.float16
        elif compute_dtype_4bit == "fp32":
            compute_dtype_4bit = torch.float32
        elif compute_dtype_4bit == "fp64":
            compute_dtype_4bit = torch.float64
        else:
            raise Exception("Invalid compute dtype for 4bit quant")

        logging.info("Checking quant dtype and load in 8bit")
        if quant_dtype_4bit and load_in_8bit:
            raise Exception("8bit quant only supports int8, fp4 can only be used with 4bit quant")

        logging.info("Checking load in 4bit and 8bit")
        if load_in_4bit and load_in_8bit:
            raise Exception("Cannot load in both 4bit and 8bit")

        logging.info("Setting BitsAndBytesConfig")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_dtype_4bit,
            bnb_4bit_compute_dtype=torch.float16,
        )

        logging.info("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        logging.info("Creating text generation pipeline")
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            device=device,
        )

        logging.info("Creating HuggingFacePipeline")
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        logging.info("Combining tools")
        tools = stock_data_tools + stock_business_tools

        logging.info("Creating conversational memory")
        conversational_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=message_history,
            return_messages=True
        )

        logging.info("Initializing agent")
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

        logging.info("Initialization successful")
        return {"message": "Initialization successful"}

    except Exception as e:
        logging.error(f"Error in test endpoint: {e}")
        return {"error": str(e)}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
