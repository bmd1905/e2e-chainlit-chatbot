from typing import Dict, Optional

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

from services.chatbot_service import ChatbotService
from services.utils import model_list

workflow_mapping = {
    "Simple Chatbot": "simple_chatbot",
    "Multi-Step Agent": "multi_step_agent",
    "Prompt Optimization": "prompt_optimization",
    "Web Search": "web_search",
}


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


@cl.set_chat_profiles
async def chat_profile():
    """
    Chat profile.
    """
    return [
        cl.ChatProfile(
            name="Simple Chatbot",
            markdown_description="This is a simple chatbot that can answer questions.",
            # icon="./assets/agent_types/chat.png",
        ),
        cl.ChatProfile(
            name="Multi-Step Agent",
            markdown_description="This is a multi-step agent that can answer questions.",
            # icon="./assets/agent_types/layer-icon.png",
        ),
        cl.ChatProfile(
            name="Prompt Optimization",
            markdown_description="This is a prompt optimization workflow that can optimize prompts.",
            # icon="./assets/agent_types/sparkle.png",
        ),
        cl.ChatProfile(
            name="Web Search",
            markdown_description="This is a web search agent that can search the internet for answers.",
            # icon="./assets/agent_types/search.png",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """
    On chat start.
    """
    # Create the chatbot service
    app = ChatbotService()

    # Set the chatbot service in the user session
    cl.user_session.set("app", app)

    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"You are now chatting with the {chat_profile} workflow."
    ).send()

    # Set the workflow type in the user session
    cl.user_session.set("workflow_type", workflow_mapping[chat_profile])

    # Setup settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=list(model_list),
                initial_value="llama-3.1-70b"
                if "llama-3.1-70b" in model_list
                else model_list[0],
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()

    # Update the model variable in the user session
    cl.user_session.set("model", settings.get("Model"))


# This receives updates in settings
@cl.on_settings_update
async def update_model(settings):
    cl.user_session.set("model", settings["Model"])


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
            user_input=message.content,
            workflow_type=cl.user_session.get("workflow_type"),
        )

        # Set the output of the parent step
        parent_step.output = result

    # Send the final result back to the user
    await cl.Message(content=parent_step.output).send()
