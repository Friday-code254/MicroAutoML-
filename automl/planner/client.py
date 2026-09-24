"""
MicroAutoML-Agent — SLM Client

A lightweight HTTP client for querying local or remote SLMs (Ollama, vLLM, etc.) 
using the normalized SLMConfig.
"""
import requests
import json
import logging
from automl.core.config import SLMConfig

logger = logging.getLogger(__name__)

class SLMClient:
    """Client for generating text from Small Language Models."""
    
    def __init__(self, config: SLMConfig):
        self.config = config
        
    def generate(self, prompt: str) -> str:
        """Invokes the SLM and returns the text response."""
        if not self.config.enabled:
            return ""
            
        if self.config.provider.lower() == "ollama":
            return self._invoke_ollama(prompt)
        elif self.config.provider.lower() == "openai":
            return self._invoke_openai_compatible(prompt)
        else:
            raise ValueError(f"Unsupported SLM provider: {self.config.provider}")
            
    def _invoke_ollama(self, prompt: str) -> str:
        url = f"{self.config.endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama invocation failed: {e}")
            raise
            
    def _invoke_openai_compatible(self, prompt: str) -> str:
        url = f"{self.config.endpoint.rstrip('/')}/v1/chat/completions"
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI-compatible invocation failed: {e}")
            raise
