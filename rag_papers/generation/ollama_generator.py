"""
Ollama generator adapter for local LLM inference.

Implements BaseGenerator interface for Ollama REST API.
"""

import requests
import time
import json
from typing import Optional, Iterator
from rag_papers.generation.generator import (
    BaseGenerator, 
    GeneratedResponse, 
    GenerationConfig
)


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
    
    def is_available(self) -> bool:
        """Check if the generator is available and ready to use."""
        return self._check_availability()
    
    def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> GeneratedResponse:
        """
        Generate text using Ollama (non-streaming).
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            GeneratedResponse with metadata
            
        Raises:
            RuntimeError: If generation fails
        """
        start_time = time.time()
        
        # Use config if provided, else defaults
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_new_tokens if config else 512
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
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
            response_text = result.get("response", "").strip()
            processing_time = time.time() - start_time
            
            return GeneratedResponse(
                text=response_text,
                confidence=0.9,  # Ollama doesn't provide confidence scores
                model_used=self.model,
                prompt_length=len(prompt),
                response_length=len(response_text),
                processing_time=processing_time
            )
            
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
    
    def stream(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Stream tokens from Ollama as they are generated.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Yields:
            Token strings as they arrive
            
        Raises:
            RuntimeError: If streaming fails
        """
        # Use config if provided, else defaults
        temperature = config.temperature if config else 0.7
        max_tokens = config.max_new_tokens if config else 512
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True  # Enable response streaming
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API returned status {response.status_code}"
                )
            
            # Parse NDJSON stream
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        
                        # Check if done
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
            
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama streaming timed out after {self.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Ollama streaming failed: {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Ollama streaming: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Ollama streaming: {str(e)}")
    
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

