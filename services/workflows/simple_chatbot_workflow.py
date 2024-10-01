import chainlit as cl
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.workflow import Event, step

from .. import logger
from ..base_workflow import BaseWorkflow
from ..schema import MessageRole


class SimpleChatbotWorkflow(BaseWorkflow):
    """
    Simple chatbot workflow that manages user interactions and generates responses
    using a language model. It maintains a chat history in the user's session.
    """

    def __init__(self, timeout: int = 60, verbose: bool = True):
        """
        Initialize the SimpleChatbotWorkflow.

        Args:
            timeout (int): The maximum time to wait for a response before timing out.
            verbose (bool): Flag to enable verbose logging.
        """
        super().__init__(timeout=timeout, verbose=verbose)

    def _setup_chat_store(self):
        """
        Setup the chat store to store the chat history in the user's session.

        This method initializes the chat store and memory buffer for the user session,
        allowing the chatbot to retain context across interactions.
        """
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
                content="You are a helpful assistant.",
            )
        )

    @cl.step(type="llm")
    @step
    async def generate_response(self, event: Event) -> Event:
        """
        Generate a response to the user's input.

        Args:
            event (Event): The event containing the user's input.

        Returns:
            Event: The event containing the generated response.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        logger.info(f"Using model: {cl.user_session.get('model')}")
        logger.info(f"History: {cl.user_session.get('memory').get()}")

        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=cl.user_session.get("memory").get(),
        )

        current_step.output = response.choices[0].message.content
        logger.info(f"Response: {current_step.output}")

        return Event(payload=current_step.output)

    async def execute_request_workflow(
        self,
        user_input: str,
    ) -> str:
        """
        Execute the request workflow.

        This method processes the user's input, generates a response, and updates
        the chat memory with both the user input and the assistant's response.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            str: The generated response from the assistant.
        """
        # Setup the chat store if it doesn't exist
        if not cl.user_session.get("memory"):
            self._setup_chat_store()

        try:
            # Add the user input to the memory
            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.HUMAN, content=user_input)
            )

            # Generate the response
            event = await self.generate_response(Event(payload=user_input))
            response = event.payload

            # Add the response to the memory
            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."
