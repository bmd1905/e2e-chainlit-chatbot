from enum import Enum


class MessageRole(Enum):
    """
    Enum for the role of the message.
    """

    HUMAN = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
