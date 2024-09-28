import asyncio

from services.chatbot_service import ChatbotService

chatbot_service = ChatbotService()


async def main():
    response = await chatbot_service.process_request(
        "Hi", "simple", model="llama-3.1-8b-instant"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
