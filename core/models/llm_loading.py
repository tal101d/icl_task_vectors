import os
from typing import Literal, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Base configuration for model loading
BASE_KWARGS = {
    "torch_dtype": torch.float16,
    "trust_remote_code": True,  # Use cautiously; only with trusted sources
}

def _setup_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    """ Set padding and EOS token for the tokenizer. """
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

def llama_local_path(variant: Literal["huggingface", "vicuna"], size: Literal["7B", "13B", "30B", "65B"]) -> str:
    """ Construct the path for locally stored Llama models. """
    llama_dir = os.environ.get("LLAMA_DIR", "./llama")
    return f"{llama_dir}/{variant}/{size}"

def get_model_path(model_type: str, model_variant: str) -> str:
    """ Retrieve the predefined path for models based on type and variant. """
    return MODEL_PATHS[model_type][model_variant]

def load_model(model_type: str, model_variant: str) -> PreTrainedModel:
    """ Load a model from a pre-trained model path and ensure it is placed on GPU if available. """
    model_path = get_model_path(model_type, model_variant)
    kwargs = BASE_KWARGS
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    return model.eval()

def load_tokenizer(model_type: str, model_variant: str) -> PreTrainedTokenizer:
    """ Load a tokenizer corresponding to a specific model. """
    model_path = get_model_path(model_type, model_variant)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    _setup_tokenizer(tokenizer)
    return tokenizer

def load_model_and_tokenizer(model_type: str, model_variant: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """ Load both model and tokenizer based on type and variant. """
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
