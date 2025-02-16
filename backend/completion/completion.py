import os

from dotenv import load_dotenv
from openai import OpenAI
from together import Together
from typing import Optional, Dict, Any

load_dotenv()

class Completer:
    """
    Basic chat completion.
    """
    def __init__(self, provider: str = "openai"):
        """
        Initialize the chat completion client.
        
        Args:
            provider (str): The LLM provider to use ('openai' or 'together')
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL_NAME")
        elif self.provider == "together":
            self.client = Together()
            self.model = os.getenv("TOGETHER_MODEL_NAME")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def complete(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Dict[str, Any]
    ) -> str:
        """
        Generate a chat completion response.
        
        Args:
            query (str): The user's input query
            system_prompt (Optional[str]): System prompt to guide the model's behavior
            temperature (float): Sampling temperature (0.0 to 1.0)
            max_tokens (int): Maximum tokens in the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            str: The model's response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content