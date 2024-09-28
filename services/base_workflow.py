from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Type

from llama_index.core.workflow import Workflow
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

available_models = [
    "llama-3.1-70b-versatile",
    "gpt-4o",
    "gpt-4o-mini",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash",
]


# Create a custom metaclass that combines WorkflowMeta and ABCMeta
class WorkflowABCMeta(type(Workflow), ABCMeta):
    pass


class BaseWorkflow(Workflow, ABC, metaclass=WorkflowABCMeta):
    def __init__(self, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = None  # Initialize llm as None

    def set_model(self, model: str):
        # Update the LLM based on the model name
        model_mapping: Dict[str, Type] = {
            "llama-3.1-70b-versatile": Groq,
            "llama-3.1-8b-instant": Groq,
            "gpt-4o": OpenAI,
            "gpt-4o-mini": OpenAI,
            "models/gemini-1.5-pro": Gemini,
            "models/gemini-1.5-flash": Gemini,
        }

        if model in model_mapping:
            self.llm = model_mapping[model](model=model)
        else:
            raise ValueError(f"Unsupported model: {model}")

    @abstractmethod
    async def execute_request_workflow(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        model: str = "llama-3.1-70b-versatile",
    ) -> str:
        self.set_model(model)  # Set the model before executing the workflow
        pass
