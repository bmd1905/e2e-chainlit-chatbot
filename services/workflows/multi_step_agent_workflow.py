from typing import Dict

import chainlit as cl
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.workflow import Event, step
from pydantic import BaseModel

from .. import logger
from ..base_workflow import BaseWorkflow
from ..schema import MessageRole


class Subtask(BaseModel):
    """
    Subtask model.
    """

    description: str
    result: str = ""


class AgentResponse(BaseModel):
    """
    Agent response model.
    """

    final_response: str
    subtask_results: Dict[str, str]


class MultiStepAgentWorkflow(BaseWorkflow):
    """
    Multi-step agent workflow that manages user interactions and generates responses
    using a language model. It maintains a chat history in the user's session.
    """

    def __init__(self, timeout: int = 60, verbose: bool = True):
        """
        Initialize the multi-step agent workflow.
        """
        super().__init__(timeout=timeout, verbose=verbose)

    def _setup_chat_store(self):
        chat_store = SimpleChatStore()

        cl.user_session.set(
            "memory",
            ChatMemoryBuffer.from_defaults(
                token_limit=10_000,
                chat_store=chat_store,
                chat_store_key=cl.user_session.get("id"),
            ),
        )

        logger.info(f"Setup chat store for user {cl.user_session.get('id')}")

        cl.user_session.get("memory").put(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful multi-step agent assistant.",
            )
        )

    @cl.step(type="llm")
    @step
    async def decompose_task(self, event: Event) -> Event:
        """
        Decompose the task into subtasks.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        decomposition_prompt = f"Break down the following user request into a maximum of 3 clear, actionable, and self-contained subtasks:\n{event.payload}"
        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=[
                *cl.user_session.get("memory").get(),
                {"role": "user", "content": decomposition_prompt},
            ],
        )

        subtasks = [
            Subtask(description=task.strip())
            for task in response.choices[0].message.content.split("\n")
            if task.strip()
        ]

        current_step.output = str(subtasks)
        return Event(payload=subtasks)

    @cl.step(type="llm")
    @step
    async def execute_subtasks(self, event: Event) -> Event:
        """
        Execute the subtasks.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        subtasks = event.payload
        for subtask in subtasks:
            execution_prompt = f"Perform the following subtask and provide a detailed result:\n{subtask.description}"
            response = await self.client.chat.completions.create(
                model=cl.user_session.get("model"),
                messages=[
                    *cl.user_session.get("memory").get(),
                    {"role": "user", "content": execution_prompt},
                ],
            )
            subtask.result = response.choices[0].message.content

        current_step.output = str(subtasks)
        return Event(payload=subtasks)

    @cl.step(type="llm")
    @step
    async def combine_results(self, event: Event) -> Event:
        """
        Combine the results of the subtasks.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        subtasks = event.payload
        combination_prompt = "Synthesize the following subtask results into a comprehensive and well-structured response:\n"
        for subtask in subtasks:
            combination_prompt += (
                f"\nSubtask: {subtask.description}\nResult: {subtask.result}\n"
            )

        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=[
                *cl.user_session.get("memory").get(),
                {"role": "user", "content": combination_prompt},
            ],
        )

        final_response = response.choices[0].message.content
        subtask_results = {subtask.description: subtask.result for subtask in subtasks}

        current_step.output = final_response
        return Event(
            payload=AgentResponse(
                final_response=final_response, subtask_results=subtask_results
            )
        )

    async def execute_request_workflow(
        self,
        user_input: str,
    ) -> str:
        """
        Execute the request workflow.
        """
        if not cl.user_session.get("memory"):
            self._setup_chat_store()

        try:
            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.HUMAN, content=user_input)
            )

            subtasks_event = await self.decompose_task(Event(payload=user_input))
            executed_subtasks_event = await self.execute_subtasks(subtasks_event)
            final_response_event = await self.combine_results(executed_subtasks_event)
            response = final_response_event.payload.final_response

            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."
