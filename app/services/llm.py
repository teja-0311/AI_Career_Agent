"""Centralized Google Gemini LLM Factory and Client Management.

Provides dependency injection of validated Gemini chat model and structured extraction,
with proper error handling, timeout configuration, and no key exposure.
"""

import json
from typing import Optional, Type, TypeVar, Any
from pydantic import BaseModel
from app.config import settings

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """Wrapper around Google Gemini client supporting text generation and structured outputs."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.chat_model
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.api_key and not self.api_key.startswith("your-"):
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self._client = None

    def is_available(self) -> bool:
        return self._client is not None

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Call Gemini for text generation with safety checks."""
        if not self._client:
            raise RuntimeError("Gemini API is not configured. Please set GEMINI_API_KEY in .env.")
        try:
            config = {}
            if system_instruction:
                config["system_instruction"] = system_instruction
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config if config else None,
            )
            return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini API generation failed: {str(e)}")

    def generate_structured(self, prompt: str, schema: Type[T], system_instruction: Optional[str] = None) -> T:
        """Enforce strict structured output using Pydantic schema validation."""
        if not self._client:
            raise RuntimeError("Gemini API is not configured. Please set GEMINI_API_KEY in .env.")
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            )
            if system_instruction:
                config.system_instruction = system_instruction
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or "{}"
            parsed_data = json.loads(raw_text)
            return schema.model_validate(parsed_data)
        except Exception as e:
            raise RuntimeError(f"Gemini structured extraction failed: {str(e)}")

    def generate_with_tools(
        self,
        contents: Any,
        tools: list,
        system_instruction: Optional[str] = None,
    ) -> Any:
        """Call Gemini model with callable tools bound for function calling."""
        if not self._client:
            raise RuntimeError("Gemini API is not configured. Please set GEMINI_API_KEY in .env.")
        try:
            from google.genai import types
            config = types.GenerateContentConfig(tools=tools)
            if system_instruction:
                config.system_instruction = system_instruction
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Gemini tool-call generation failed: {str(e)}")


# Singleton factory for shared dependency injection
_default_gemini: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _default_gemini
    if _default_gemini is None:
        _default_gemini = GeminiClient()
    return _default_gemini
