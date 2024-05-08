import os
from typing import Literal, Tuple, Optional
import logging
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Setup logging
# Create a basic logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define a formatter including the date and time
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Assign the formatter to the handler
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


# Base configuration for model loading
BASE_KWARGS = {
    "torch_dtype": torch.float16,
    "trust_remote_code": True,  # Use cautiously; only with trusted sources
}

def _setup_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    """Set padding and EOS token for the tokenizer."""
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

def llama_local_path(variant: Literal["huggingface", "vicuna"], size: Literal["7B", "13B", "30B", "65B"]) -> Optional[str]:
    """Construct the path for locally stored Llama models."""
    llama_dir = os.environ.get("LLAMA_DIR", "./llama")
    path = f"{llama_dir}/{variant}/{size}"
    if os.path.exists(path):
        return path
    logging.error(f"Llama model path does not exist: {path}")
    return None

def get_model_path(model_type: str, model_variant: str) -> str:
    """Retrieve the predefined path for models based on type and variant."""
    try:
        return MODEL_PATHS[model_type][model_variant]
    except KeyError as e:
        logging.error(f"Model type or variant not found: {e}")
        raise

def load_model(model_type: str, model_variant: str) -> Optional[PreTrainedModel]:
    """Load a model from a pre-trained model path and ensure it is placed on GPU if available."""
    try:
        model_path = get_model_path(model_type, model_variant)
        model = AutoModelForCausalLM.from_pretrained(model_path, **BASE_KWARGS)
        # Move model to GPU if available
        if torch.cuda.is_available():
            model.cuda()
        return model.eval()
    except Exception as e:
        logging.error(f"Error loading model {model_type}, {model_variant}: {e}")
        raise

def load_tokenizer(model_type: str, model_variant: str) -> Optional[PreTrainedTokenizer]:
    """Load a tokenizer corresponding to a specific model."""
    try:
        model_path = get_model_path(model_type, model_variant)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        _setup_tokenizer(tokenizer)
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer for {model_type}, {model_variant}: {e}")
        raise

def load_model_and_tokenizer(model_type: str, model_variant: str) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
    """Load both model and tokenizer based on type and variant."""
    tokenizer = load_tokenizer(model_type, model_variant)
    model = load_model(model_type, model_variant)
    return model, tokenizer

# Model paths configuration
MODEL_PATHS = {
    "pythia": {
        "1.4B": "EleutherAI/pythia-1.4b",
        "2.8B": "EleutherAI/pythia-2.8b",
        "6.9B": "EleutherAI/pythia-6.9b",
        "12B": "EleutherAI/pythia-12b",
    },
    "falcon": {
        "7B": "tiiuae/falcon-7b",
        "40B": "tiiuae/falcon-40b",
    },
    "gpt-j": {
        "6B": "EleutherAI/gpt-j-6B",
    },
    "gpt-2": {
        "0.35B": "gpt2-medium",
        "0.77B": "gpt2-large",
        "1.5B": "gpt2-xl",
    },
    "mpt": {
        "7B": "mosaicml/mpt-7b",
    },
    "gpt-neox": {
        "20B": "EleutherAI/gpt-neox-20b",
    },
    "starcoder": {
        "regular": "bigcode/starcoder",
        "plus": "bigcode/starcoderplus",
    },
    "cerebras-gpt": {
        "6.7B": "cerebras/Cerebras-GPT-6.7B",
        "13B": "cerebras/Cerebras-GPT-13B",
    }
}
