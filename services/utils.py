from typing import Dict, List

import httpx

from . import logger


# Load model configurations from the API
def load_model_configurations() -> List[Dict[str, str]]:
    """
    Load model configurations from the API.

    :return: List[Dict[str, str]]: The list of model configurations.
    """
    with httpx.Client() as client:
        response = client.get(
            "http://0.0.0.0:4000/models", headers={"accept": "application/json"}
        )
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()["data"]  # Return the list of models

        model_list = [model["id"] for model in data]

        return model_list


# Load the model mapping from the API
model_list = load_model_configurations()

logger.info(f"Available models: {model_list}")
