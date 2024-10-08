import os
from typing import Annotated, Any, List, Optional, Dict

import chainlit as cl
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import LLM
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.groq import Groq
from llama_index.tools.tavily_research import TavilyToolSpec

llm = Groq(model="llama-3.1-70b-versatile")

### Define tools
search_tool_spec = TavilyToolSpec(api_key=os.getenv("TAVILY"))
search_tools = search_tool_spec.to_tool_list()


### Define events
class SearchEvent(Event):
    """Requires the LLM to do an online search to answer the question"""

    query: Annotated[str, "The user's query"]


class AnswerEvent(Event):
    """Allows the LLM to answer the question without searching"""

    query: Annotated[str, "The user's query"]


class ResponseEvent(Event):
    """Collects LLM response"""

    query: Annotated[str, "The user's query"]
    answer: Annotated[str, "The LLM's response"]


### Define workfow
class MixtureOfAnswers(Workflow):
    def __init__(self, *args: Any, llm: Optional[LLM] = llm, **kwargs: Any):
        """Class constructor. Takes in an llm instance and constructs
        1. A function calling agent with search tools
        2. A simple chat engine instance
        3. A common memory instance across the workflow

        Args:
            llm (Optional[LLM], optional): LLM instance. Defaults to Settings.llm.
        """
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.search_agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=search_tools, llm=self.llm
        )
        self.search_agent = self.search_agent_worker.as_agent()
        self.answer_without_search_engine = SimpleChatEngine.from_defaults(llm=self.llm)
        self.history: List[ChatMessage] = []

    @cl.step(type="llm")
    @step()
    async def route_to_llm(self, ev: StartEvent) -> SearchEvent | AnswerEvent:
        """Generates a search event and an answer event once given a start event"""

        ## Update memory
        self.history.append(ChatMessage(role=MessageRole.USER, content=ev.query))

        ## Routes to both events. But you can also write a router component to decide
        ## which event to route to.
        self.send_event(SearchEvent(query=ev.query))
        self.send_event(AnswerEvent(query=ev.query))

    @cl.step(type="tool")
    @step()
    async def search_and_answer(self, ev: SearchEvent) -> ResponseEvent:
        """Uses the tavily search tool to answer the question"""

        ## Synthesize response
        response = await self.search_agent.achat(ev.query, chat_history=self.history)

        ## [OPTIONAL] Show intermediate response in the frontend
        await cl.Message(content="ANSWER WITH SEARCH: " + str(response)).send()

        ## Update memory
        self.history.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="ANSWER WITH SEARCH: " + str(response),
            )
        )

        return ResponseEvent(query=ev.query, answer=str(response))

    @cl.step(type="llm")
    @step()
    async def simply_answer(self, ev: AnswerEvent) -> ResponseEvent:
        """Uses the LLM to simple answer the question"""

        ## Synthesize response
        response = await self.answer_without_search_engine.achat(
            ev.query, chat_history=self.history
        )

        ## [OPTIONAL] Show intermediate response in the frontend
        await cl.Message(content="ANSWER WITHOUT SEARCH: " + str(response)).send()

        ## Update memory
        self.history.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="ANSWER WITHOUT SEARCH: " + str(response),
            )
        )

        return ResponseEvent(query=ev.query, answer=str(response))

    @cl.step(type="llm")
    @step()
    async def compile(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        """Compiles and summarizes answers from all response events"""

        ## There are 2 response events from routing to 2 different agents. This can
        ## also be a dynamic number of events.
        ready = ctx.collect_events(ev, [ResponseEvent] * 2)

        if ready is None:
            return None

        response = await self.llm.acomplete(
            f"""
            A user has asked us a question and we have responded accordingly using a
            search tool and without using a search tool. Your job is to decide which
            response best answered the question and summarize the response into a crisp
            reply. If both responses answered the question, summarize both responses
            into a single answer.

            The user's query was: {ev.query}

            The responses are:
            {ready[0].answer} &
            {ready[1].answer}
            """
        )

        ## Update memory
        self.history.append(
            ChatMessage(
                role=MessageRole.ASSISTANT, content="FINAL ANSWER: " + str(response)
            )
        )

        return StopEvent(result=str(response))

    async def execute_request_workflow(
        self, user_input: str, history: List[Dict[str, str]] = None
    ) -> str:
        """Executes the workflow based on user input and conversation history."""
        try:
            # Convert history to ChatMessage objects and add to memory
            if history:
                for message in history:
                    role = message.get("role", "").upper()
                    if role == "USER":
                        role = MessageRole.USER
                    elif role == "ASSISTANT":
                        role = MessageRole.ASSISTANT
                    else:
                        role = MessageRole.SYSTEM
                    self.history.append(
                        ChatMessage(role=role, content=message.get("content", ""))
                    )

            # Add the current user input to history
            self.history.append(ChatMessage(role=MessageRole.USER, content=user_input))

            # Run the app logic
            result = await self.run(query=user_input)

            return result

        except Exception as e:
            # Handle exceptions
            return f"Error processing request: {str(e)}"


### Define the app - with just a few lines of code
@cl.on_chat_start
async def on_chat_start():
    app = MixtureOfAnswers(
        verbose=True, timeout=6000
    )  # The app times out if it runs for 6000s without any result
    cl.user_session.set("app", app)
    await cl.Message("Hello! Ask me anything!").send()


@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("app")

    # Start a parent step
    async with cl.Step(name="Processing Query") as parent_step:
        parent_step.input = message.content

        # Run the execute_request_workflow logic
        result = await app.execute_request_workflow(user_input=message.content)

        # Set the output of the parent step
        parent_step.output = result

    # Send the final result back to the user
    await cl.Message(content=parent_step.output).send()
