from services.workflows.multi_step_agent_workflow import MultiStepAgentWorkflow
from services.workflows.prompt_optimization_workflow import PromptOptimizationWorkflow
from services.workflows.simple_chatbot_workflow import SimpleChatbotWorkflow

from .base_workflow import BaseWorkflow


class WorkflowFactory:
    @staticmethod
    def create_workflow(workflow_type: str, **kwargs) -> BaseWorkflow:
        if workflow_type == "prompt_optim":
            return PromptOptimizationWorkflow(**kwargs)
        elif workflow_type == "multi_step":
            return MultiStepAgentWorkflow(**kwargs)
        elif workflow_type == "simple":
            return SimpleChatbotWorkflow(**kwargs)
        else:
            raise ValueError(f"Invalid workflow type: {workflow_type}")
