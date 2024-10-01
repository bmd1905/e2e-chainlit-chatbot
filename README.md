# E2E Chainlit Chatbot

This application manages various chatbot services using Chainlit and LlamaIndex. It provides a flexible framework for building and deploying chatbots with advanced capabilities.

## Features

- **Multi-step workflows**: Supports complex interactions with users through multi-step workflows, allowing for more engaging conversations.
- **Prompt optimization**: Automatically optimizes user prompts for better responses, enhancing the overall user experience.
- **Chat history management**: Maintains a history of interactions for context-aware responses, ensuring continuity in conversations.
- **Customizable models**: Easily switch between different LLM models to tailor the chatbot's behavior to specific needs.
- **Web Search**: Integrates a web search capability using the Tavily search tool to provide real-time answers based on user queries.
- **Authentication**: Supports user authentication via GitHub and Google, allowing for secure access to the chatbot services.

## Setup

To set up the project, follow these steps:

1. **Clone the repository**:

```bash
git clone https://github.com/bmd1905/e2e-chainlit-chatbot.git
cd e2e-chainlit-chatbot
```

2. **Install the required dependencies**:

```bash
# Create and activate environment
conda create -n e2e-chainlit-chatbot python=3.11 -y
conda activate e2e-chainlit-chatbot

# Install libraries
pip install -r requirements.txt
```

3. **Install pre-commit dependencies**:

```bash
pre-commit install
```

## Usage
Start LiteLLM proxy server with the following command:

```bash
make litellm
```

Then you can start the chatbot service using the following command:

```bash
make cl
```

5. **Access the chatbot**:
Open your browser and navigate to `http://localhost:8000` to interact with your chatbot.

## Usage

Once the application is running, you can start chatting with the bot. It will respond based on the configured workflows and models.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. Make sure to follow the coding standards and include tests for any new features.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
