from typing import Dict, List

from llama_index.core.llms import ChatMessage

from .workflows.multi_step_agent_workflow import MessageRole


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
