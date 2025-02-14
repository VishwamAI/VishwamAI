#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Building and testing VishwamAI Ollama model..."

# Check prerequisites
command -v ollama >/dev/null 2>&1 || { echo -e "${RED}Error: Ollama is not installed${NC}" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}Error: Python3 is not installed${NC}" >&2; exit 1; }

# Convert model to Ollama format
echo -e "\n${GREEN}Converting model to Ollama format...${NC}"
python3 convert_to_ollama.py --input-dir ../final_model --output-dir ./ollama_model

# Check if conversion was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Model conversion failed${NC}"
    exit 1
fi

# Create Ollama model
echo -e "\n${GREEN}Creating Ollama model...${NC}"
ollama create vishwamai -f Modelfile

# Check if model creation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Ollama model creation failed${NC}"
    exit 1
fi

# Test the model
echo -e "\n${GREEN}Testing model with sample prompts...${NC}"

test_model() {
    local prompt="$1"
    echo -e "\nTesting prompt: ${prompt}"
    ollama run vishwamai "${prompt}" || {
        echo -e "${RED}Error: Model test failed${NC}"
        return 1
    }
}

# Run test cases
test_prompts=(
    "What is deep learning?"
    "Write a Python function to calculate Fibonacci numbers."
    "Explain the concept of neural networks."
)

failed_tests=0
for prompt in "${test_prompts[@]}"; do
    if ! test_model "$prompt"; then
        ((failed_tests++))
    fi
done

# Print test results
echo -e "\n${GREEN}Test Results:${NC}"
echo "Total tests: ${#test_prompts[@]}"
echo "Failed tests: ${failed_tests}"

# Test memory and cache components
echo -e "\n${GREEN}Testing model components...${NC}"
python3 - <<EOF
import torch
from pathlib import Path
from vishwamai.model import Transformer, ModelArgs

def test_components():
    try:
        # Load model config
        config_path = Path("./ollama_model/config.json")
        if not config_path.exists():
            raise FileNotFoundError("Model config not found")
            
        # Test model loading
        print("Testing model loading...")
        model = Transformer(ModelArgs.from_json(config_path))
        assert model is not None, "Failed to load model"
        
        # Test inference
        print("Testing inference...")
        with torch.inference_mode():
            test_input = torch.randint(0, 32000, (1, 32)).cuda()
            output = model(test_input)
            assert output is not None, "Model inference failed"
            
        print("Component tests passed successfully!")
        return 0
    except Exception as e:
        print(f"Component test failed: {str(e)}")
        return 1

exit(test_components())
EOF

component_test_result=$?

# Final status
if [ $failed_tests -eq 0 ] && [ $component_test_result -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed successfully!${NC}"
    echo -e "Model is ready for use. Try: ollama run vishwamai"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Please check the logs above.${NC}"
    exit 1
fi
