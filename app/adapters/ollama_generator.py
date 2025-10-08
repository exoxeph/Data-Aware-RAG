"""
Ollama generator adapter for local LLM inference.

Implements BaseGenerator interface for Ollama REST API.
"""

import requests
from typing import Optional
from rag_papers.generation.generator import BaseGenerator


class OllamaGenerator(BaseGenerator):
    """
    Generator that uses Ollama for local LLM inference.
    
    Ollama must be running locally (default: http://localhost:11434).
    Supports streaming or non-streaming generation.
    """
    
    def __init__(
        self, 
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 60
    ):
        """
        Initialize Ollama generator.
        
        Args:
            model: Ollama model name (e.g., "llama3", "mistral", "phi")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            
        Raises:
            RuntimeError: If Ollama server is not reachable
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # Check server availability
        if not self._check_availability():
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url}. "
                f"Please start Ollama with 'ollama serve' or check the URL."
            )
    
    def _check_availability(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: int = 512
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If generation fails
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,  # Non-streaming for simplicity
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API returned status {response.status_code}: {response.text}"
                )
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                f"Try a shorter prompt or increase timeout."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Ollama request failed: {str(e)}. "
                f"Check if Ollama is running at {self.base_url}."
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Ollama generation: {str(e)}")
    
    def get_available_models(self) -> list[str]:
        """
        Get list of available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception:
            return []
    
    def __repr__(self) -> str:
        return f"OllamaGenerator(model='{self.model}', base_url='{self.base_url}')"
