"""Model implementation for evaluation"""

import os
from openai import AsyncOpenAI
from typing import Any, Dict, List, Optional
from inspect_ai.model import Model as BaseModel
from .config import Config

class Model(BaseModel):
    """Model implementation for evaluation"""

    def __init__(self, model: str, config: Optional[Config] = None):
        self.original_model = model  # Keep original for reference
        self.config = config or Config()
        self.api = self  # For compatibility with base class
        
        # Initialize OpenAI client with flexible base URL and API routing
        api_key = None
        base_url = None
        
        # Determine which API to use based on model name
        # Models with provider prefixes (openai/, anthropic/, google/, etc.) should use OpenRouter
        if model.startswith("openrouter/"):
            # Explicit openrouter/ prefix
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            self.model = model.replace("openrouter/", "")
            print(f"[MODEL] Using OpenRouter with model: {self.model}")
        elif "/" in model and model.split("/")[0] in ["openai", "anthropic", "google", "meta-llama", "microsoft", "mistralai", "cohere"]:
            # Auto-detect OpenRouter from provider prefix
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            self.model = model
            print(f"[MODEL] Using OpenRouter (auto-detected) with model: {self.model}")
        else:
            # Direct API (OpenAI, NVIDIA, etc.) for plain model names
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            self.model = model
            print(f"[MODEL] Using direct API at {base_url} with model: {self.model}")
        
        if not api_key:
            raise ValueError(f"API key not found. For model '{model}', please set the appropriate API key in your .env file")
        
        print(f"[MODEL] API key found: {api_key[:15]}...")
        print(f"[MODEL] Base URL: {base_url}")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def max_connections(self):
        return 1

    async def generate(self, state: Any) -> Any:
        # Get the prompt from the state
        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'content'):
                prompt = last_message.content
            else:
                prompt = str(last_message)
        else:
            prompt = ""

        # Convert state messages to OpenAI format
        messages = []
        for msg in state.messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                messages.append({
                    "role": "user" if len(messages) % 2 == 0 else "assistant",
                    "content": str(msg)
                })

        # Add system message for structured output
        messages.insert(0, {
            "role": "system",
            "content": """You are a helpful AI assistant. Please structure your responses in the following format:

ANALYSIS: Break down what is being asked

CONSIDERATIONS:
- Key point 1
- Key point 2
- Key point 3
- Key point 4

APPROACH:
1. Step one
2. Step two
3. Step three
4. Step four

RESPONSE:
Detailed response with specific actions and plans."""
        })

        # Call OpenAI API
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            response = await completion
            response_text = response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Model API: {str(e)}")
            response_text = "I apologize, but I encountered an error. Please ensure your Model API key is set correctly."

        # Create a new state with the response
        from .task_state import TaskState
        from .message import Message
        return TaskState(
            model=state.model,
            sample_id=state.sample_id,
            epoch=state.epoch,
            input=state.input,
            messages=state.messages + [
                Message(
                    role="assistant",
                    content=response_text
                )
            ],
            metadata=state.metadata
        )
