import chainlit as cl

from services.chatbot_service import ChatbotService


@cl.on_chat_start
async def on_chat_start():
    app = ChatbotService()
    cl.user_session.set("app", app)
    await cl.Message("Hello! Ask me anything!").send()


@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("app")

    # Start a parent step
    async with cl.Step(name="Processing Query") as parent_step:
        parent_step.input = message.content

        # Run the execute_request_workflow logic
        result = await app.process_request(
            message.content,
            "multi_step_agent",
            model="llama-3.1-70b-versatile",
        )

        # Set the output of the parent step
        parent_step.output = result

    # Send the final result back to the user
    await cl.Message(content=parent_step.output).send()
