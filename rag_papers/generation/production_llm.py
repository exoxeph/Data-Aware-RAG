"""
Real LLM Integration for Production Deployment

This module provides production-ready integrations with various LLM providers
including OpenAI, Azure OpenAI, and Anthropic for real-world deployment.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from .generator import BaseGenerator, GeneratedResponse, GenerationConfig


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str  # "openai", "azure", "anthropic"
    api_key: str
    model_name: str
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None  # For Azure
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 1000


class ProductionLLMGenerator(BaseGenerator):
    """Production LLM generator with multiple provider support."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider."""
        if self.config.provider == "openai":
            return self._init_openai_client()
        elif self.config.provider == "azure":
            return self._init_azure_client()
        elif self.config.provider == "anthropic":
            return self._init_anthropic_client()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except ImportError:
            raise ImportError("OpenAI package required: pip install openai")
    
    def _init_azure_client(self):
        """Initialize Azure OpenAI client."""
        try:
            import openai
            return openai.AzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version or "2023-12-01-preview",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except ImportError:
            raise ImportError("OpenAI package required: pip install openai")
    
    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            return anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except ImportError:
            raise ImportError("Anthropic package required: pip install anthropic")
    
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        try:
            # Simple health check
            if self.config.provider in ["openai", "azure"]:
                # For OpenAI/Azure, we could make a minimal completion call
                return True
            elif self.config.provider == "anthropic":
                # For Anthropic, check client initialization
                return self.client is not None
            return False
        except Exception as e:
            self.logger.warning(f"LLM availability check failed: {e}")
            return False
    
    def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> GeneratedResponse:
        """Generate response using the configured LLM provider."""
        gen_config = config or GenerationConfig()
        
        try:
            if self.config.provider in ["openai", "azure"]:
                return self._generate_openai(prompt, gen_config)
            elif self.config.provider == "anthropic":
                return self._generate_anthropic(prompt, gen_config)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            # Fallback to error response
            return GeneratedResponse(
                text=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                model_used=f"{self.config.provider}_{self.config.model_name}",
                confidence=0.0,
                prompt_length=len(prompt),
                metadata={"error": str(e)}
            )
    
    def _generate_openai(
        self, 
        prompt: str, 
        config: GenerationConfig
    ) -> GeneratedResponse:
        """Generate response using OpenAI/Azure OpenAI."""
        messages = [{"role": "user", "content": prompt}]
        
        # Use deployment name for Azure, model name for OpenAI
        model = self.config.deployment_name or self.config.model_name
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=min(config.max_new_tokens, self.config.max_tokens),
            top_p=config.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract response text
        content = response.choices[0].message.content
        
        # Calculate confidence based on response characteristics
        confidence = self._calculate_confidence(content, response)
        
        return GeneratedResponse(
            text=content,
            model_used=f"{self.config.provider}_{model}",
            confidence=confidence,
            prompt_length=len(prompt),
            metadata={
                "usage": asdict(response.usage) if response.usage else {},
                "finish_reason": response.choices[0].finish_reason
            }
        )
    
    def _generate_anthropic(
        self, 
        prompt: str, 
        config: GenerationConfig
    ) -> GeneratedResponse:
        """Generate response using Anthropic Claude."""
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=min(config.max_new_tokens, self.config.max_tokens),
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract response text
        content = response.content[0].text
        
        # Calculate confidence
        confidence = self._calculate_confidence(content, response)
        
        return GeneratedResponse(
            text=content,
            model_used=f"anthropic_{self.config.model_name}",
            confidence=confidence,
            prompt_length=len(prompt),
            metadata={
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                } if response.usage else {},
                "stop_reason": response.stop_reason
            }
        )
    
    def _calculate_confidence(self, content: str, response: Any) -> float:
        """Calculate confidence score based on response characteristics."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on response length
        if len(content) < 50:
            confidence -= 0.2
        elif len(content) > 500:
            confidence += 0.1
        
        # Adjust based on finish reason (OpenAI/Azure)
        if hasattr(response, 'choices') and response.choices:
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "stop":
                confidence += 0.1
            elif finish_reason == "length":
                confidence -= 0.1
        
        # Adjust based on stop reason (Anthropic)
        if hasattr(response, 'stop_reason'):
            if response.stop_reason == "end_turn":
                confidence += 0.1
            elif response.stop_reason == "max_tokens":
                confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))


class LLMFactory:
    """Factory for creating LLM generators with different configurations."""
    
    @staticmethod
    def create_openai_generator(
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        **kwargs
    ) -> ProductionLLMGenerator:
        """Create OpenAI generator."""
        config = LLMConfig(
            provider="openai",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            **kwargs
        )
        return ProductionLLMGenerator(config)
    
    @staticmethod
    def create_azure_generator(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        deployment_name: str = "gpt-35-turbo",
        api_version: str = "2023-12-01-preview",
        **kwargs
    ) -> ProductionLLMGenerator:
        """Create Azure OpenAI generator."""
        config = LLMConfig(
            provider="azure",
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            model_name=deployment_name,
            base_url=base_url or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version,
            deployment_name=deployment_name,
            **kwargs
        )
        return ProductionLLMGenerator(config)
    
    @staticmethod
    def create_anthropic_generator(
        api_key: Optional[str] = None,
        model_name: str = "claude-3-sonnet-20240229",
        **kwargs
    ) -> ProductionLLMGenerator:
        """Create Anthropic generator."""
        config = LLMConfig(
            provider="anthropic",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model_name=model_name,
            **kwargs
        )
        return ProductionLLMGenerator(config)


# Async version for high-throughput applications
class AsyncProductionLLMGenerator(BaseGenerator):
    """Async version of production LLM generator for high-throughput applications."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None  # Will be initialized in async context
        self.logger = logging.getLogger(__name__)
    
    async def ainit(self):
        """Async initialization."""
        self.client = await self._initialize_async_client()
    
    async def _initialize_async_client(self):
        """Initialize async LLM client."""
        if self.config.provider in ["openai", "azure"]:
            try:
                import openai
                if self.config.provider == "openai":
                    return openai.AsyncOpenAI(
                        api_key=self.config.api_key,
                        timeout=self.config.timeout,
                        max_retries=self.config.max_retries
                    )
                else:  # Azure
                    return openai.AsyncAzureOpenAI(
                        api_key=self.config.api_key,
                        azure_endpoint=self.config.base_url,
                        api_version=self.config.api_version,
                        timeout=self.config.timeout,
                        max_retries=self.config.max_retries
                    )
            except ImportError:
                raise ImportError("OpenAI package required: pip install openai")
        else:
            raise ValueError(f"Async not supported for provider: {self.config.provider}")
    
    def is_available(self) -> bool:
        """Check if the async LLM service is available."""
        return self.client is not None
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GeneratedResponse:
        """Sync wrapper for async generation."""
        return asyncio.run(self.agenerate(prompt, config))
    
    async def agenerate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> GeneratedResponse:
        """Async generate response."""
        if not self.client:
            await self.ainit()
        
        gen_config = config or GenerationConfig()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            model = self.config.deployment_name or self.config.model_name
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=gen_config.temperature,
                max_tokens=min(gen_config.max_new_tokens, self.config.max_tokens),
                top_p=gen_config.top_p
            )
            
            content = response.choices[0].message.content
            confidence = 0.8  # Simplified confidence calculation
            
            return GeneratedResponse(
                text=content,
                model_used=f"{self.config.provider}_{model}",
                confidence=confidence,
                prompt_length=len(prompt),
                metadata={
                    "usage": asdict(response.usage) if response.usage else {},
                    "finish_reason": response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            self.logger.error(f"Async generation failed: {e}")
            return GeneratedResponse(
                text=f"Error: {str(e)}",
                model_used=f"{self.config.provider}_error",
                confidence=0.0,
                prompt_length=len(prompt),
                metadata={"error": str(e)}
            )


# Usage examples and integration
def create_production_rag_generator(
    provider: str = "openai",
    **kwargs
) -> 'RAGGenerator':
    """Create a production RAG generator with real LLM integration."""
    from .generator import RAGGenerator
    
    if provider == "openai":
        llm_generator = LLMFactory.create_openai_generator(**kwargs)
    elif provider == "azure":
        llm_generator = LLMFactory.create_azure_generator(**kwargs)
    elif provider == "anthropic":
        llm_generator = LLMFactory.create_anthropic_generator(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return RAGGenerator(generator=llm_generator)


# Configuration helpers
def load_llm_config_from_env() -> Dict[str, Any]:
    """Load LLM configuration from environment variables."""
    config = {}
    
    # OpenAI configuration
    if os.getenv("OPENAI_API_KEY"):
        config["openai"] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "base_url": os.getenv("OPENAI_BASE_URL")
        }
    
    # Azure OpenAI configuration
    if os.getenv("AZURE_OPENAI_API_KEY"):
        config["azure"] = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        }
    
    # Anthropic configuration
    if os.getenv("ANTHROPIC_API_KEY"):
        config["anthropic"] = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model_name": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        }
    
    return config