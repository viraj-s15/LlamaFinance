# Stock GPT

## Table of Contents

- [Stock GPT](#stock-gpt)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Prerequisites](#prerequisites)
    - [Installing Dependencies for development](#installing-dependencies-for-development)
  - [Usage ](#usage-)
    - [General](#general)
    - [Command Line Arguments](#command-line-arguments)
    - [Other helpful commands](#other-helpful-commands)
  - [Contributing](#contributing)
  - [Attribution](#attribution)

## About <a name = "about"></a>

A Custom Chatgpt with Multiple Stock based Tools at its disposal

## Getting Started <a name = "getting_started"></a>

### Prerequisites

If you want to use GPT, make sure you change the name of the `src/openai/.example_env` to `src/openai/.env` and add your Openai API key

### Installing Dependencies for development

Downloading dependencies for AMD Systems

```
bash scripts/dev_setup_amd.sh
```

Downloading dependencies for Nvidia Systems

```
bash scripts/dev_setup_nvidia.sh
```

Downloading dependencies For GPT related development 

```
bash scripts/dev_setup_openai.sh
```

You now have your dependencies setup

## Usage <a name = "usage"></a>

```
>>> Give me the current stock price of TSLA
The current stock price of TSLA is $243.84.
```

### General<a name = "general"></a>

One script will setup the dependencies and get you up and running.

```
bash scripts/usage/run_openai.sh
```

### Command Line Arguments<a name = "command-line-arguments"></a>

(Do not touch these if you dont know what youre doing)

- --verbose : Its presence sets the LLM output to verbose
- --temperature : Sets the temprature of the LLM, default value is 0 
- --model: The GPT model you want to use
- --max_iterations: The maximum number of iterations the model is allowed to make
- --message_history: The number of messages to store in the conversation history

### Other helpful commands
- exit : Exits the chat application
- help : Gives information of all available options for running the script

## Contributing

Refer <a href="">our contribution docs</a>

## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
