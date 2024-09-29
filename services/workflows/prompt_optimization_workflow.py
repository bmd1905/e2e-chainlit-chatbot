import asyncio
from enum import Enum
from typing import Dict, List, Optional

import chainlit as cl
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Event, StartEvent, StopEvent, step
from pydantic import BaseModel

from .. import logger
from ..base_workflow import BaseWorkflow


class MessageRole(Enum):
    HUMAN = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class OptimizePromptEvent(Event):
    optimized_prompt: str


class GenerateResponseEvent(Event):
    final_prompt: str


class EvaluatePromptOutput(BaseModel):
    needs_optimization: bool


class OptimizePromptOutput(BaseModel):
    optimized_prompt: str


class PromptOptimizationWorkflow(BaseWorkflow):
    # Prompt templates
    evaluation_prompt_template = PromptTemplate(
        "Evaluate the following user prompt to determine if optimization is needed, "
        "considering the conversation history. "
        "User Prompt: {user_prompt}\nConversation History: {history}"
    )

    optimization_prompt_template = PromptTemplate(
        "Improve the following user prompt to better fit the entire conversation history:\n"
        "Original Prompt: {original_prompt}\nConversation History: {history}"
    )

    def __init__(self, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=1024)

    @cl.step(type="llm")
    @step
    async def evaluate_prompt(
        self, event: StartEvent
    ) -> GenerateResponseEvent | OptimizePromptEvent:
        current_step = cl.context.current_step

        current_step.input = event.user_prompt

        # Evaluate the user prompt
        evaluation_response = await self.llm.astructured_predict(
            output_cls=EvaluatePromptOutput,
            prompt=self.evaluation_prompt_template,
            user_prompt=event.user_prompt,
            history=event.get("history", ""),
        )

        needs_optimization = evaluation_response.needs_optimization

        logger.info(f"Is optimization needed: {needs_optimization}")

        current_step.output = str(needs_optimization)

        if needs_optimization:
            return OptimizePromptEvent(optimized_prompt=event.user_prompt)

        return GenerateResponseEvent(final_prompt=event.user_prompt)

    @cl.step(type="llm")
    @step
    async def optimize_prompt(
        self, event: OptimizePromptEvent
    ) -> GenerateResponseEvent:
        current_step = cl.context.current_step

        current_step.input = event.optimized_prompt

        # Optimize the user prompt
        optimization_response = await self.llm.astructured_predict(
            output_cls=OptimizePromptOutput,
            prompt=self.optimization_prompt_template,
            original_prompt=event.optimized_prompt,
            history=event.get("history", ""),
        )
        optimized_prompt = optimization_response.optimized_prompt

        logger.info(f"Optimized Prompt: {optimized_prompt}")

        current_step.output = optimized_prompt

        return GenerateResponseEvent(final_prompt=optimized_prompt)

    @cl.step(type="llm")
    @step
    async def generate_response(self, event: GenerateResponseEvent) -> StopEvent:
        current_step = cl.context.current_step

        current_step.input = event.final_prompt

        # Generate the chatbot's response
        response_prompt = f"Chatbot response to: {event.final_prompt}"
        chatbot_response = await self.llm.acomplete(response_prompt)

        current_step.output = str(chatbot_response).strip()
        return StopEvent(result=str(chatbot_response).strip())

    async def execute_request_workflow(
        self, user_input: str, history: List[Dict[str, str]] = None, model: str = ...
    ) -> str:
        logger.info(f"Model: {model}")
        self.set_model(model)  # Set the model before executing the workflow
        try:
            # Convert history to ChatMessage objects and add to memory
            if history:
                for message in history:
                    role = message.get("role", "").upper()
                    if role == "USER":
                        role = MessageRole.HUMAN
                    elif role == "ASSISTANT":
                        role = MessageRole.ASSISTANT
                    else:
                        role = MessageRole.SYSTEM
                    self.memory.put(
                        ChatMessage(role=role, content=message.get("content", ""))
                    )

            # Add the current user input to memory
            self.memory.put(ChatMessage(role=MessageRole.HUMAN, content=user_input))

            # Get the chat history as a string
            chat_history = "\n".join(
                [f"{msg.role.value}: {msg.content}" for msg in self.memory.get()]
            )

            logger.info(f"Chat History: {chat_history}")

            # Evaluate the prompt
            event = await self.evaluate_prompt(
                StartEvent(user_prompt=user_input, history=chat_history)
            )
            if isinstance(event, OptimizePromptEvent):
                # Optimize the prompt if needed
                event = await self.optimize_prompt(event)

            # Generate the final response
            response_event = await self.generate_response(event)
            final_response = response_event.result

            # Add the final response to memory
            self.memory.put(
                ChatMessage(role=MessageRole.ASSISTANT, content=final_response)
            )

            return final_response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."


async def main():
    # Initialize the workflow
    chatbot_workflow = PromptOptimizationWorkflow(timeout=60, verbose=True)

    # Example user prompt
    user_prompt = "MLOps?"

    # Example conversation history
    conversation_history: Optional[List[Dict[str, str]]] = [
        {"role": "user", "content": "Hi, I need help with machine learning."},
        {
            "role": "assistant",
            "content": "Sure, I'd be happy to help you with Machine Learning.",
        },
    ]

    logger.info(f"User: {user_prompt}")

    # Run the workflow with history
    final_response = await chatbot_workflow.execute_request_workflow(
        user_input=user_prompt,
        history=conversation_history,
    )

    # Print the chatbot's response
    print(f"Chatbot: {final_response}")


if __name__ == "__main__":
    asyncio.run(main())
