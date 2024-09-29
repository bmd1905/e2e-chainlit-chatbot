# Chainlit Chatbot Service

This application manages various chatbot services using Chainlit and LlamaIndex. It provides a flexible framework for building and deploying chatbots with advanced capabilities.

## Features

- **Multi-step workflows**: Supports complex interactions with users through multi-step workflows, allowing for more engaging conversations.
- **Prompt optimization**: Automatically optimizes user prompts for better responses, enhancing the overall user experience.
- **Chat history management**: Maintains a history of interactions for context-aware responses, ensuring continuity in conversations.
- **Customizable models**: Easily switch between different LLM models to tailor the chatbot's behavior to specific needs.

## Setup

To set up the project, follow these steps:

1. **Clone the repository**:

```bash
git clone https://github.com/bmd1905/chainlit.git
cd chainlit
```

2. **Install the required dependencies**:

```bash
pip install -r requirements.txt
```

3. **Install pre-commit dependencies**:

```bash
pre-commit install
```

4. **Run the application**:
You can start the chatbot service using the following command:

```bash
python main.py
```

5. **Access the chatbot**:
Open your browser and navigate to `http://localhost:3000` to interact with your chatbot.

## Usage

Once the application is running, you can start chatting with the bot. It will respond based on the configured workflows and models.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. Make sure to follow the coding standards and include tests for any new features.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
