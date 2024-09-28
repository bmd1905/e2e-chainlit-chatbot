from enum import Enum
from typing import Dict, List

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Event, step

from .. import logger
from ..base_workflow import BaseWorkflow


class MessageRole(Enum):
    HUMAN = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SimpleChatbotWorkflow(BaseWorkflow):
    def __init__(self, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=1024)

    @step
    async def generate_response(self, event: Event) -> Event:
        user_input = event.payload
        chat_history = "\n".join(
            [f"{msg.role.value}: {msg.content}" for msg in self.memory.get()]
        )

        prompt = f"Given the following conversation history:\n{chat_history}\n\nUser: {user_input}\nAssistant:"
        response = await self.llm.acomplete(prompt)
        return Event(payload=str(response).strip())

    async def execute_request_workflow(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        model: str = "",
    ) -> str:
        logger.info(f"Model: {model}")
        self.set_model(model)  # Set the model before executing the workflow
        try:
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

            self.memory.put(ChatMessage(role=MessageRole.HUMAN, content=user_input))

            event = await self.generate_response(Event(payload=user_input))
            response = event.payload

            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response))

            return response

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."


async def main():
    chatbot_workflow = SimpleChatbotWorkflow(timeout=60, verbose=True)
    user_prompt = "Tell me a joke about programming."
    conversation_history = [
        {"role": "user", "content": "Hi, how are you?"},
        {
            "role": "assistant",
            "content": "Hello! I'm doing well, thank you for asking. How can I assist you today?",
        },
    ]

    logger.info(f"User: {user_prompt}")

    final_response = await chatbot_workflow.execute_request_workflow(
        user_input=user_prompt,
        history=conversation_history,
    )

    print(f"Chatbot: {final_response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
