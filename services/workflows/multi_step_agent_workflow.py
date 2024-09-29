import asyncio
from enum import Enum
from typing import Dict, List

import chainlit as cl
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Event, step
from pydantic import BaseModel, Field

from .. import logger
from ..base_workflow import BaseWorkflow
from ..utils import store_history_in_memory


class MessageRole(Enum):
    """
    Enum for the role of the message.
    """

    HUMAN = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Subtask(BaseModel):
    """
    Subtask model.
    """

    description: str
    result: str = ""


class AgentRequest(BaseModel):
    """
    Agent request model.
    """

    user_input: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    subtasks: List[Subtask] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """
    Agent response model.
    """

    final_response: str
    subtask_results: Dict[str, str]


class SubtasksOut(BaseModel):
    """
    Subtasks output model.
    """

    subtasks: List[str] = Field(
        ..., description="List of subtasks to complete the user request."
    )


class MultiStepAgentWorkflow(BaseWorkflow):
    """
    Multi-step agent workflow.
    """

    # Prompt templates
    decomposition_prompt_template = PromptTemplate(
        "Break down the following user request into a maximum of 3 clear, actionable, and self-contained subtasks. "
        "Each subtask should represent a logical step towards fulfilling the request and have a specific, measurable outcome. "
        "Consider the different components or stages involved in completing the request. "
        "Provide sufficient context and instructions for each subtask to be executed independently:\n{user_input}"
    )

    execution_prompt_template = PromptTemplate(
        "Perform the following subtask and provide a detailed result. "
        "Ensure the response is clear and directly addresses the task:\n{subtask_description}"
    )

    combination_prompt_template = PromptTemplate(
        "Synthesize the following subtask results into a comprehensive and well-structured response. "
        "Ensure a smooth flow of information and logical transitions between the different sections. "
        "Address all subtasks in a coherent manner, maintaining clarity and conciseness. "
        "The final output should read as a unified whole, not a collection of separate parts:\n{subtask_results}"
    )

    final_response_prompt_template = PromptTemplate(
        "Refine the following draft response into a polished and natural-sounding final answer. "
        "Focus on clarity, conciseness, and a smooth, engaging writing style. "
        "Ensure the response is easy to understand and free of any awkward phrasing or grammatical errors. "
        "Maintain the original meaning and information while enhancing the overall quality of the writing:\n{draft_response}"
    )

    def __init__(self, timeout: int = 60, verbose: bool = True):
        """
        Initialize the multi-step agent workflow.
        """
        super().__init__(timeout=timeout, verbose=verbose)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=1024)

    @cl.step(type="llm")
    @step
    async def decompose_task(self, event: Event) -> Event:
        """
        Decompose the task into subtasks.
        """
        current_step = cl.context.current_step

        current_step.input = event.payload.user_input

        # Decompose the task into subtasks
        request = event.payload
        response = await self.llm.astructured_predict(
            output_cls=SubtasksOut,
            prompt=self.decomposition_prompt_template,
            user_input=request.user_input,
        )

        # Create subtasks
        subtasks = [
            Subtask(description=task.strip())
            for task in response.subtasks
            if task.strip()
        ]
        request.subtasks = subtasks

        current_step.output = str(subtasks)

        return Event(payload=request)

    @step
    async def execute_subtasks(self, event: Event) -> Event:
        """
        Execute the subtasks.
        """
        current_step = cl.context.current_step

        current_step.input = event.payload

        # Execute the subtasks
        request = event.payload

        async def execute_single_subtask(subtask: Subtask):
            response = await self.llm.acomplete(
                self.execution_prompt_template.format(
                    subtask_description=subtask.description
                )
            )
            subtask.result = str(response).strip()

        await asyncio.gather(
            *(execute_single_subtask(subtask) for subtask in request.subtasks)
        )

        # Set the output of the current step
        current_step.output = str(request.subtasks)

        return Event(payload=request)

    @cl.step(type="llm")
    @step
    async def combine_results(self, event: Event) -> Event:
        """
        Combine the results of the subtasks.
        """
        current_step = cl.context.current_step

        current_step.input = event.payload

        request = event.payload

        # Combine the results of the subtasks
        subtask_results = {
            subtask.description: subtask.result for subtask in request.subtasks
        }
        response = await self.llm.acomplete(
            self.combination_prompt_template.format(subtask_results=subtask_results)
        )

        # Set the output of the current step
        current_step.output = str(response)

        return Event(
            payload=AgentResponse(
                final_response=str(response).strip(), subtask_results=subtask_results
            )
        )

    @cl.step(type="llm")
    @step
    async def generate_final_response(self, event: Event) -> Event:
        """
        Generate the final response.
        """
        current_step = cl.context.current_step

        current_step.input = event.payload

        response = event.payload

        # Generate the final response
        final_response = await self.llm.acomplete(
            self.final_response_prompt_template.format(
                draft_response=response.final_response
            )
        )
        response.final_response = str(final_response).strip()

        # Set the output of the current step
        current_step.output = str(response)

        return Event(payload=response)

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

        # Execute the workflow
        try:
            # Convert history to ChatMessage objects and add to memory
            store_history_in_memory(history, self.memory)

            # Add the current user input to memory
            self.memory.put(ChatMessage(role=MessageRole.HUMAN, content=user_input))

            # Task Decomposition
            event = await self.decompose_task(
                Event(payload=AgentRequest(user_input=user_input))
            )
            request = event.payload
            logger.info(f"Subtasks: {request.subtasks}")

            # Parallel Execution
            event = await self.execute_subtasks(Event(payload=request))
            request = event.payload

            # Result Combination
            event = await self.combine_results(Event(payload=request))
            response = event.payload

            # Final Response Generation
            event = await self.generate_final_response(Event(payload=response))
            response = event.payload

            # Add the final response to memory
            self.memory.put(
                ChatMessage(role=MessageRole.ASSISTANT, content=response.final_response)
            )

            return response.final_response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."
