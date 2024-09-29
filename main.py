import chainlit as cl

from services.chatbot_service import ChatbotService


@cl.on_chat_start
async def on_chat_start():
    """
    On chat start.
    """
    # Create the chatbot service
    app = ChatbotService()

    # Set the chatbot service in the user session
    cl.user_session.set("app", app)

    # Send a welcome message to the user
    await cl.Message("Hello! Ask me anything!").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    On message.
    """
    # Get the app from the user session
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
