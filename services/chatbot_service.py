from typing import Dict, List

from .workflow_factory import WorkflowFactory


class ChatbotService:
    def __init__(self):
        self.workflow_factory = WorkflowFactory()

    async def process_request(
        self,
        user_input: str,
        workflow_type: str,
        history: List[Dict[str, str]] = None,
        model: str = "llama-3.1-70b-versatile",
    ) -> str:
        workflow = self.workflow_factory.create_workflow(workflow_type)
        return await workflow.execute_request_workflow(user_input, history, model)
