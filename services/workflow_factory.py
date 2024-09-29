from services.workflows.multi_step_agent_workflow import MultiStepAgentWorkflow
from services.workflows.prompt_optimization_workflow import PromptOptimizationWorkflow
from services.workflows.simple_chatbot_workflow import SimpleChatbotWorkflow
from services.workflows.web_search_workflow import WebSearchWorkflow

from .base_workflow import BaseWorkflow


class WorkflowFactory:
    """
    Workflow factory.
    """

    @staticmethod
    def create_workflow(workflow_type: str, **kwargs) -> BaseWorkflow:
        """
        Create a workflow based on the workflow type.
        """
        if workflow_type == "prompt_optimization":
            return PromptOptimizationWorkflow(**kwargs)
        elif workflow_type == "multi_step_agent":
            return MultiStepAgentWorkflow(**kwargs)
        elif workflow_type == "simple_chatbot":
            return SimpleChatbotWorkflow(**kwargs)
        elif workflow_type == "web_search":
            return WebSearchWorkflow(**kwargs)
        else:
            raise ValueError(f"Invalid workflow type: {workflow_type}")
