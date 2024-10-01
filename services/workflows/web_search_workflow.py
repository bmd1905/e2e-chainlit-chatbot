import os

import chainlit as cl
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.workflow import Event, step
from llama_index.llms.groq import Groq
from llama_index.tools.tavily_research import TavilyToolSpec

from .. import logger
from ..base_workflow import BaseWorkflow
from ..schema import MessageRole


class WebSearchWorkflow(BaseWorkflow):
    """
    WebSearchWorkflow orchestrates a web search process using the Tavily search tool.

    This workflow manages user interactions and generates responses by leveraging a language model
    and web search capabilities. It maintains a chat history within the user's session, allowing
    for context-aware responses.

    Attributes:
        search_tool_spec (TavilyToolSpec): Specification for the Tavily search tool.
        search_tools (list): List of tools derived from the Tavily specification.
        llm (Groq): Language model used for generating responses.

    Args:
        timeout (int): Maximum time (in seconds) to wait for a response before timing out.
        verbose (bool): Flag to enable verbose logging for debugging purposes.
    """

    def __init__(self, timeout: int = 60, verbose: bool = True):
        """
        Initializes the WebSearchWorkflow with specified timeout and verbosity.

        Args:
            timeout (int): Maximum time (in seconds) to wait for a response before timing out.
            verbose (bool): Flag to enable verbose logging for debugging purposes.
        """
        super().__init__(timeout=timeout, verbose=verbose)
        self.search_tool_spec = TavilyToolSpec(api_key=os.getenv("TAVILY"))
        self.search_tools = self.search_tool_spec.to_tool_list()

        self.llm = Groq(
            model="llama-3.1-70b-versatile",
        )

    def _setup_chat_store(self):
        """
        Initializes the chat store for the user session.

        This method sets up a SimpleChatStore and associates it with the user's session memory,
        allowing for the storage of chat history and context.
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

    @cl.step(type="llm")
    @step
    async def optimize_query(self, event: Event) -> Event:
        """
        Optimizes the user query for web searching.

        This method takes the user's input query and reformulates it into a concise,
        search-friendly format. It utilizes the language model to generate the optimized query.

        Args:
            event (Event): The event containing the user input.

        Returns:
            Event: An event containing the optimized query.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        # Create a prompt to optimize the user query
        optimization_prompt = f"Convert the following user query into a concise, search-friendly format suitable for web searching:\nUser Query: {event.payload}"

        # Call the language model to get the optimized query
        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=[
                *cl.user_session.get("memory").get(),
                {"role": "user", "content": optimization_prompt},
            ],
        )

        optimized_query = response.choices[0].message.content
        current_step.output = optimized_query
        return Event(payload=optimized_query)

    @cl.step(type="tool")
    @step
    async def perform_web_search(self, event: Event) -> Event:
        """
        Performs a web search using the optimized query.

        This method utilizes the FunctionCallingAgent to execute the search with the provided
        tools and returns the search results.

        Args:
            event (Event): The event containing the optimized query.

        Returns:
            Event: An event containing the search results.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        # Create a search agent using the specified tools and language model
        search_agent = FunctionCallingAgent.from_tools(
            tools=self.search_tools,
            llm=self.llm,
            verbose=True,
            allow_parallel_tool_calls=True,
            system_prompt="You are a helpful assistant that can perform web searches.",
        )

        # Execute the search and get results
        search_results = await search_agent.achat(event.payload)

        current_step.output = str(search_results)
        return Event(payload=str(search_results))

    @cl.step(type="llm")
    @step
    async def generate_final_response(self, event: Event) -> Event:
        """
        Generates the final response based on search results.

        This method formulates a comprehensive answer to the user's query by incorporating
        the search results. It utilizes the language model to create the final response.

        Args:
            event (Event): The event containing the search results.

        Returns:
            Event: An event containing the final response.
        """
        current_step = cl.context.current_step
        current_step.input = event.payload

        # Create a prompt to generate the final response
        response_prompt = f"Based on the following search results, provide a comprehensive and informative answer to the user's query, including relevant URLs:\nSearch Results: {event.payload}\nFinal Response:"

        # Call the language model to get the final response
        response = await self.client.chat.completions.create(
            model=cl.user_session.get("model"),
            messages=[
                *cl.user_session.get("memory").get(),
                {"role": "user", "content": response_prompt},
            ],
        )

        final_response = response.choices[0].message.content
        current_step.output = final_response
        return Event(payload=final_response)

    async def execute_request_workflow(
        self,
        user_input: str,
    ) -> str:
        """
        Executes the entire request workflow.

        This method orchestrates the process of optimizing the user query, performing a web search,
        and generating the final response. It also manages the chat memory for the user session.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            str: The final response generated after processing the user input.
        """
        if not cl.user_session.get("memory"):
            self._setup_chat_store()

        try:
            # Store the user's input in memory
            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.HUMAN, content=user_input)
            )

            # Optimize the user query
            optimized_query_event = await self.optimize_query(Event(payload=user_input))
            # Perform the web search
            search_results_event = await self.perform_web_search(optimized_query_event)
            # Generate the final response
            final_response_event = await self.generate_final_response(
                search_results_event
            )
            response = final_response_event.payload

            # Store the assistant's response in memory
            cl.user_session.get("memory").put(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."
