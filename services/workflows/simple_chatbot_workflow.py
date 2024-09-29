from typing import Dict, List

import chainlit as cl
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Event, step

from .. import logger
from ..base_workflow import BaseWorkflow
from ..schema import MessageRole
from ..utils import store_history_in_memory


class SimpleChatbotWorkflow(BaseWorkflow):
    """
    Simple chatbot workflow.
    """

    def __init__(self, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=1024)

    @cl.step(type="llm")
    @step
    async def generate_response(self, event: Event) -> Event:
        """
        Generate a response to the user's input.
        """
        current_step = cl.context.current_step

        current_step.input = event.payload

        # Get the chat history as a string
        chat_history = "\n".join(
            [f"{msg.role.value}: {msg.content}" for msg in self.memory.get()]
        )

        prompt = f"Given the following conversation history:\n{chat_history}\n\nUser: {current_step.input}\nAssistant:"
        response = await self.llm.acomplete(prompt)

        current_step.output = str(response).strip()

        return Event(payload=current_step.output)

    async def execute_request_workflow(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        model: str = "",
    ) -> str:
        """
        Execute the request workflow.
        """
        logger.info(f"Model: {model}")
        self.set_model(model)  # Set the model before executing the workflow

        # Add the history to the memory using the utility function
        try:
            store_history_in_memory(history, self.memory)

            self.memory.put(ChatMessage(role=MessageRole.HUMAN, content=user_input))

            # Generate the response
            event = await self.generate_response(Event(payload=user_input))
            response = event.payload

            # Add the response to the memory
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response))

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."
