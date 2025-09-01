# PDF-based QA

## Prerequisites

- Pyenv: Choose which Python version you’re using.
- Poetry: Manage a project’s dependencies and create a venv with that Python version.

## Getting Started

### Pyenv Setup

1. Install Pyenv:

```
brew update
brew install pyenv
```

2. Command on the terminal

```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

3. Restart the terminal:

```
exec "$SHELL"
```

4. Install Python:

```
pyenv install 3.11
pyenv global 3.11
exec zsh
# check python version
python --version
```

### Poetry Setup

- Install Poetry:

```
pip3 install poetry
poetry env activate
```

- Install dependencies:

```
poetry install
```

- Batch update of Python package:

```
poetry update
```

### Environment Setup

1. Create a `.env` file in the root directory with the following variables

```
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=your_project_name
```

### IDE Setup

To resolve import issues in VS Code:

1. Activate Poetry environment:

```
poetry shell
```

2. Or set Python interpreter to Poetry virtual environment:
   - Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose the Poetry virtual environment path (e.g., `/Users/.../pypoetry/virtualenvs/rag-xxx-py3.11/bin/python`)

### Streamlit Setup

- Run the application:

```
streamlit run PDF-based_QA/main.py
```

## What did we use RAG?

1️⃣ You can save the document in the internal DB and accumulate the contents.
2️⃣ Write an answer based on the content of the document → Return to the document and verify the answered content.
3️⃣ You can expect better answer quality (low halulation), and it is possible to create a domain-specific chatbot that answers based on extensive knowledge.
