#!/bin/bash

# Check GPU
if lspci | grep -i NVIDIA > /dev/null; then
  GPU="Nvidia"
  SCRIPT="dev_setup_nvidia.sh"
elif lspci | grep -i AMD > /dev/null; then
  GPU="AMD"
  SCRIPT="dev_setup_amd.sh"
else
  echo "No compatible GPU found. Exiting."
  exit 1
fi

# Install GPU-specific setup script
bash scripts/development/$SCRIPT

# Ask user for LLM type
read -p "Do you want to run the full LLM or only the quantised version? (full/quantised): " llm_type

if [[ $llm_type == "quantised" ]]; then
  # Ask user for quantised type
  read -p "Which type of quantised version do you want to run? (awq/bnb): " quant_type

  if [[ $quant_type == "awq" ]]; then
    echo "To run the quantised version, use the following command:"
    echo "python src/transformers/transformers_quantised_awq.py --verbose --temperature 0.1 --model mistralai/Mistral-7B-Instruct-v0.1 --max_iterations 3 --message_history 5 --cache_dir None --max-new-tokens 512 --repetition-penalty 1.1 --load_in_4bit --quant_dtype_4bit fp4 --compute_dtype_4bit fp16 --device 0"
    echo ""
  elif [[ $quant_type == "bnb" ]]; then
    echo "To run the quantised version, use the following command:"
    echo "python src/transformers/transformers_quantised_bnb.py --verbose --temperature 0.1 --model mistralai/Mistral-7B-Instruct-v0.1 --max_iterations 3 --message_history 5 --cache_dir None --max-new-tokens 512 --repetition-penalty 1.1 --load_in_4bit --quant_dtype_4bit fp4 --compute_dtype_4bit fp16 --device 0"
    echo ""
  else
    echo "Invalid quantised type. Exiting."
    exit 1
  fi
else
  echo "To run the full LLM, use the following command:"
  echo "python src/transformers/transformers_llm.py --verbose --temperature 0.1 --model mistralai/Mistral-7B-Instruct-v0.1 --max_iterations 3 --message_history 5 --cache_dir None --max-new-tokens 512 --repetition-penalty 1.1 --load_in_4bit --quant_dtype_4bit fp4 --compute_dtype_4bit fp16 --device 0"
  echo ""
  echo "Explanation of flags:"
  echo "--verbose: Enable verbose mode for detailed output."
  echo "--temperature: Set the sampling temperature."
  echo "--model: Specify the LLM model to use."
  echo "--max_iterations: Set the maximum number of iterations."
  echo "--message_history: Set the number of previous messages to consider."
  echo "--cache_dir: Specify the directory for caching."
  echo "--max-new-tokens: Set the maximum number of new tokens to generate."
  echo "--repetition-penalty: Set the repetition penalty."
  echo "--load_in_4bit: Load the model in 4-bit quantized format."
  echo "--quant_dtype_4bit: Specify the data type for 4-bit quantization."
  echo "--compute_dtype_4bit: Specify the data type for computation in 4-bit mode."
  echo "--device: Specify the device to use (e.g., GPU index)."
fi
