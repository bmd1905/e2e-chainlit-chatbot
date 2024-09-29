from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List

from llama_index.core.workflow import Workflow

from .utils import model_mapping


# Create a custom metaclass that combines WorkflowMeta and ABCMeta
class WorkflowABCMeta(type(Workflow), ABCMeta):
    pass


class BaseWorkflow(Workflow, ABC, metaclass=WorkflowABCMeta):
    """
    Base workflow class.
    """

    def __init__(self, timeout: int = 60, verbose: bool = True):
        """
        Initialize the BaseWorkflow.
        """
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = None  # Initialize llm as None

    def set_model(self, model: str):
        """
        Set the model based on the model name.
        """
        if model in model_mapping:
            model_class_path = model_mapping[model]
            module_name, class_name = model_class_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            self.llm = model_class(model=model)
        else:
            raise ValueError(f"Unsupported model: {model}")

    @abstractmethod
    async def execute_request_workflow(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        model: str = "llama-3.1-70b-versatile",
    ) -> str:
        """
        Execute the request workflow.
        """
        self.set_model(model)  # Set the model before executing the workflow
        pass
