# Stock GPT

## Table of Contents

- [Stock GPT](#stock-gpt)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
  - [Usage ](#usage-)
    - [General](#general)
    - [Command Line Arguments](#command-line-arguments)
  - [TODO](#todo)
    - [Completed](#completed)
  - [Contributing](#contributing)

## About <a name = "about"></a>

A Custom Chatgpt with Multiple Stock based Tools at its disposal

## Getting Started <a name = "getting_started"></a>

Make sure you have poetry installed, if youre using Arch Linux, install it with the command below

### Prerequisites

```
yay -S python-poetry
```

Make sure you change the name of the `.example_env`
to `.env` and add your Openai API key

### Installing

```
cd stock-gpt
poetry install
```

You now have your dependencies setup

## Usage <a name = "usage"></a>

### General<a name = "general"></a>

To just run the script on the terminal

```
cd openai
poetry run python main.py
```

### Command Line Arguments<a name = "command-line-arguments"></a>

(Do not touch these if you dont know what youre doing)

- --verbose : Its presence sets the LLM output to verbose
- --temperature : Sets the temprature of the LLM, default value is 0 

## TODO
- [ ] Add Support for other open source llms
- [ ] Add support for local llms (llama.cpp)
- [ ] Create demo notebook for people to use
- [ ] Create GUI for chat    

### Completed 
- [ ] None

## Contributing

Feel free to open an issue and contribute