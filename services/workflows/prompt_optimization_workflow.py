import chainlit as cl
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.workflow import Event, step

from .. import logger
from ..base_workflow import BaseWorkflow
from ..schema import MessageRole


class PromptOptimizationWorkflow(BaseWorkflow):
    """
    PromptOptimizationWorkflow orchestrates the process of optimizing user prompts
    for better interaction with a language model. It manages user interactions and
    maintains a chat history within the user's session to provide context-aware responses.
    """

    def __init__(self, timeout: int = 60, verbose: bool = True):
        """
        Initializes the PromptOptimizationWorkflow with specified timeout and verbosity.

        Args:
            timeout (int): Maximum time (in seconds) to wait for a response before timing out.
            verbose (bool): Flag to enable verbose logging for debugging purposes.
        """
        super().__init__(timeout=timeout, verbose=verbose)

    def _setup_chat_store(self):
        """
        Sets up the chat store for the user session.

        This method initializes a SimpleChatStore and associates it with the user's session memory,
        allowing for the storage of chat history and context. It also initializes the system message
        to guide the assistant's behavior.
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
                content="You are a helpful assistant that optimizes prompts.",
            )
        )

    @cl.step(type="llm")
    @step
    async def evaluate_prompt(self, event: Event) -> Event:
        """
        Evaluates the user prompt to determine if optimization is needed.

        This method constructs an evaluation prompt and sends it to the language model,
        checking if the prompt requires optimization based on the conversation history.

        Args:
            event (Event): The event containing the user prompt.

        Returns:
            Event: An event containing a boolean indicating if optimization is needed
                   and the original prompt.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        evaluation_prompt = f"Evaluate the following user prompt to determine if optimization is needed, considering the conversation history:\nUser Prompt: {event.payload}"
        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=[
                *cl.user_session.get("memory").get(),
                {"role": "user", "content": evaluation_prompt},
            ],
        )

        needs_optimization = "yes" in response.choices[0].message.content.lower()
        current_step.output = str(needs_optimization)
        return Event(
            payload={
                "needs_optimization": needs_optimization,
                "original_prompt": event.payload,
            }
        )

    @cl.step(type="llm")
    @step
    async def optimize_prompt(self, event: Event) -> Event:
        """
        Optimizes the user prompt if needed.

        This method checks if the prompt needs optimization and, if so, constructs an
        optimization prompt to improve the original user prompt based on the conversation history.

        Args:
            event (Event): The event containing the evaluation result and original prompt.

        Returns:
            Event: An event containing the optimized prompt or the original prompt if no optimization is needed.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        if not event.payload["needs_optimization"]:
            return Event(payload=event.payload["original_prompt"])

        optimization_prompt = f"Improve the following user prompt to better fit the entire conversation history:\nOriginal Prompt: {event.payload['original_prompt']}"
        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=[
                *cl.user_session.get("memory").get(),
                {"role": "user", "content": optimization_prompt},
            ],
        )

        optimized_prompt = response.choices[0].message.content
        current_step.output = optimized_prompt
        return Event(payload=optimized_prompt)

    @cl.step(type="llm")
    @step
    async def generate_response(self, event: Event) -> Event:
        """
        Generates a response based on the optimized prompt.

        This method sends the optimized prompt to the language model and retrieves the response.

        Args:
            event (Event): The event containing the optimized prompt.

        Returns:
            Event: An event containing the generated response from the language model.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=[
                *cl.user_session.get("memory").get(),
                {"role": "user", "content": event.payload},
            ],
        )

        current_step.output = response.choices[0].message.content
        return Event(payload=response.choices[0].message.content)

    async def execute_request_workflow(
        self,
        user_input: str,
    ) -> str:
        """
        Executes the entire request workflow for prompt optimization.

        This method orchestrates the process of evaluating the user prompt, optimizing it if necessary,
        and generating a response. It also manages the chat memory for the user session.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            str: The final response generated after processing the user input.
        """
        if not cl.user_session.get("memory"):
            self._setup_chat_store()

        try:
            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.HUMAN, content=user_input)
            )

            evaluation_event = await self.evaluate_prompt(Event(payload=user_input))
            optimized_prompt_event = await self.optimize_prompt(evaluation_event)
            response_event = await self.generate_response(optimized_prompt_event)
            response = response_event.payload

            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."
