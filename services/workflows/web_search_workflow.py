import os
from typing import Dict, List

import chainlit as cl
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Event, step
from llama_index.tools.tavily_research import TavilyToolSpec

from ..base_workflow import BaseWorkflow
from ..schema import MessageRole
from ..utils import store_history_in_memory


class WebSearchWorkflow(BaseWorkflow):
    """
    Web search workflow using Tavily search tool.
    """

    PROMPT_OPTIMIZE_QUERY = PromptTemplate(
        "Convert the following user query into a concise, search-friendly format suitable for web searching:"
        "\n\nUser Query: {user_query}"
    )
    PROMPT_FINAL_RESPONSE = PromptTemplate(
        "Based on the following search results, provide a comprehensive and informative answer to the user's query, including relevant URLs:"
        "\n\nSearch Results: {search_results}\n\nFinal Response:"
    )

    def __init__(self, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=1024)
        self.search_tool_spec = TavilyToolSpec(api_key=os.getenv("TAVILY"))
        self.search_tools = self.search_tool_spec.to_tool_list()

    @cl.step(type="llm")
    @step
    async def optimize_query(self, event: Event) -> Event:
        """Optimize the user query for web search."""
        current_step = cl.context.current_step
        current_step.input = event.payload

        optimized_query = await self.llm.acomplete(
            self.PROMPT_OPTIMIZE_QUERY.format(user_query=event.payload)
        )

        current_step.output = str(optimized_query).strip()
        return Event(payload=current_step.output)

    @cl.step(type="tool")
    @step
    async def perform_web_search(self, event: Event) -> Event:
        """Perform web search using Tavily search tool."""
        current_step = cl.context.current_step
        current_step.input = event.payload

        search_agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=self.search_tools, llm=self.llm
        )
        search_agent = search_agent_worker.as_agent()
        search_results = await search_agent.achat(event.payload)

        current_step.output = str(search_results)
        return Event(payload=current_step.output)

    @cl.step(type="llm")
    @step
    async def generate_final_response(self, event: Event) -> Event:
        """Generate final response based on search results."""
        current_step = cl.context.current_step
        current_step.input = event.payload

        final_response = await self.llm.acomplete(
            self.PROMPT_FINAL_RESPONSE.format(search_results=event.payload)
        )

        current_step.output = str(final_response).strip()
        return Event(payload=current_step.output)

    async def execute_request_workflow(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        model: str = "llama-3.1-70b-versatile",
    ) -> str:
        """Execute the web search workflow."""
        if not user_input:
            return "Error: User input cannot be empty."

        self.set_model(model)  # Set the model before executing the workflow

        store_history_in_memory(history, self.memory)
        self.memory.put(ChatMessage(role=MessageRole.HUMAN, content=user_input))

        # Optimize the query
        optimized_query_event = await self.optimize_query(Event(payload=user_input))
        optimized_query = optimized_query_event.payload

        # Perform web search
        search_results_event = await self.perform_web_search(
            Event(payload=optimized_query)
        )
        search_results = search_results_event.payload

        # Generate final response
        final_response_event = await self.generate_final_response(
            Event(payload=search_results)
        )
        final_response = final_response_event.payload

        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=final_response))

        return final_response
