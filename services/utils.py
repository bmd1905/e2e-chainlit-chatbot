from pathlib import Path
from typing import Dict, List

import yaml
from llama_index.core.llms import ChatMessage

from .schema import MessageRole


def store_history_in_memory(history: List[Dict[str, str]], memory) -> None:
    """
    Convert history to ChatMessage objects and add to memory.

    Args:
        history (List[Dict[str, str]]): The conversation history.
        memory: The memory object to store ChatMessage instances.
    """
    if history:
        for message in history:
            role = message.get("role", "").upper()
            if role == "USER":
                role = MessageRole.HUMAN
            elif role == "ASSISTANT":
                role = MessageRole.ASSISTANT
            else:
                role = MessageRole.SYSTEM
            memory.put(ChatMessage(role=role, content=message.get("content", "")))


# Load model configurations from YAML file
def load_model_configurations():
    """
    Load model configurations from YAML file.
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)["models"]


# Load the model mapping from the YAML file
model_mapping = load_model_configurations()
