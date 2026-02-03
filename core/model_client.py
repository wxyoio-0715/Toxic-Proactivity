"""Model client abstractions used by the misalignment simulator and workflow."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from anthropic import Anthropic, APIStatusError  # type: ignore
except ImportError:  # pragma: no cover - anthropic is optional
    Anthropic = None  # type: ignore
    APIStatusError = Exception  # type: ignore

try:
    from openai import OpenAI, APIError as OpenAIAPIError  # type: ignore
except ImportError:  # pragma: no cover - openai is optional
    OpenAI = None  # type: ignore
    OpenAIAPIError = Exception  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
    from google.api_core.exceptions import GoogleAPIError  # type: ignore
except ImportError:  # pragma: no cover - google genai is optional
    genai = None  # type: ignore

    class GoogleAPIError(Exception):  # type: ignore
        """Fallback error when google api_core is unavailable."""


LOGGER = logging.getLogger(__name__)


class ModelClient:
    """Generic LLM client wrapper with pluggable providers."""

    def __init__(
        self,
        provider: str,
        *,
        api_key: str,
        model_name: str,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        use_reasoning: bool = False,
    ) -> None:
        if not provider:
            raise ValueError("Model provider must be specified")
        if not model_name:
            raise ValueError("Model name must be specified")

        self._provider = provider.lower()
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._client = None
        self._base_url = base_url or ""
        self._api_key = api_key
        self._is_openrouter = False
        self._temperature = temperature
        self._top_p = top_p
        self._extra_body = extra_body
        self._use_reasoning = use_reasoning

        if self._provider == "openrouter":
            self._is_openrouter = True
            if not self._base_url:
                self._base_url = "https://openrouter.ai/api/v1"

        if self._provider == "anthropic":
            if not api_key:
                raise ValueError("Anthropic API key must be provided when using the anthropic provider")
            if Anthropic is None:
                raise ImportError("anthropic package is not available. Install it to enable Anthropic calls.")
            self._client = Anthropic(api_key=api_key)
        elif self._provider in {"openai", "openrouter"}:
            if not api_key:
                raise ValueError("OpenAI-compatible API key must be provided when using the openai/openrouter provider")
            if OpenAI is None:
                raise ImportError("openai package is not available. Install it to enable OpenAI calls.")
            client_kwargs = {"api_key": api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = OpenAI(**client_kwargs)
        elif self._provider == "echo":
            self._client = None
        elif self._provider == "google":
            if not api_key:
                raise ValueError("Google Generative AI API key must be provided when using the google provider")
            if genai is None:
                raise ImportError("google-generativeai package is not available. Install it to enable Google calls.")
            configure_args = {"api_key": api_key}
            if self._base_url:
                configure_args["client_options"] = {"api_endpoint": self._base_url}
            genai.configure(**configure_args)
            self._client = genai.GenerativeModel(model_name)
        elif self._provider == "deepseek":
            if not api_key:
                raise ValueError("DeepSeek API key must be provided when using the deepseek provider")
            self._client = {
                "session": requests.Session(),
                "base_url": (self._base_url.rstrip("/") or "https://api.deepseek.com"),
            }
        else:
            raise NotImplementedError(f"Unsupported model provider: {self._provider}")

    def invoke(self, prompt: str, *, max_retries: int = 3, retry_delay: float = 2.0) -> str:
        if not prompt:
            raise ValueError("Prompt must be a non-empty string")
        messages = [{"role": "user", "content": prompt}]
        return self.invoke_messages(messages, max_retries=max_retries, retry_delay=retry_delay)

    def invoke_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        if self._provider == "echo":
            return "\n".join(msg.get("content", "") for msg in messages).strip()

        attempt = 0
        last_error: Optional[Exception] = None
        temperature_value = temperature if temperature is not None else self._temperature
        top_p_value = top_p if top_p is not None else self._top_p
        max_tokens_value = max_tokens if max_tokens is not None else self._max_tokens

        while attempt < max_retries:
            attempt += 1
            try:
                LOGGER.debug(
                    "Invoking %s model '%s' (attempt %s)", self._provider, self._model_name, attempt
                )
                if self._provider == "anthropic":
                    system_prompt = ""
                    cleaned_messages = []
                    for msg in messages:
                        if msg.get("role") == "system":
                            system_prompt += (msg.get("content", "") + "\n").strip()
                        else:
                            cleaned_messages.append(
                                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                            )
                    create_kwargs: Dict[str, Any] = {
                        "model": self._model_name,
                        "messages": cleaned_messages or [{"role": "user", "content": ""}],
                        "system": system_prompt or None,
                        "max_tokens": max_tokens_value,
                        "temperature": temperature_value,
                    }
                    if top_p_value is not None:
                        create_kwargs["top_p"] = top_p_value
                    response = self._client.messages.create(  # type: ignore[union-attr]
                        **create_kwargs
                    )
                    if not getattr(response, "content", None):
                        raise RuntimeError("Model response did not contain any content segments")
                    segment = response.content[0]
                    text = getattr(segment, "text", None)
                    if not text:
                        raise RuntimeError("Model response segment did not contain text content")
                    return text.strip()

                if self._provider in {"openai", "openrouter"}:
                    create_kwargs: Dict[str, Any] = {
                        "model": self._model_name,
                        "messages": messages,
                        "max_tokens": max_tokens_value,
                        "temperature": temperature_value if temperature_value is not None else 0.7,
                    }
                    if top_p_value is not None:
                        create_kwargs["top_p"] = top_p_value
                    if self._extra_body:
                        create_kwargs["extra_body"] = self._extra_body
                    response = self._client.chat.completions.create(  # type: ignore[union-attr]
                        **create_kwargs
                    )
                    choices = getattr(response, "choices", None)
                    if not choices:
                        raise RuntimeError("Model response did not contain any choices")
                    message = getattr(choices[0], "message", None)
                    if message is None:
                        raise RuntimeError("Model response choice did not include a message")
                    content = getattr(message, "content", None)
                    reasoning = self._extract_reasoning(message)
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict):
                                text_parts.append(str(part.get("text", "")).strip())
                            else:
                                text_parts.append(str(part))
                        content = " ".join(filter(None, text_parts))
                    if not content:
                        raise RuntimeError("Model response message did not include text content")
                    content_text = str(content).strip()
                    if self._use_reasoning and reasoning:
                        content_text = self._merge_reasoning(content_text, reasoning)
                    return content_text

                if self._provider == "google":
                    flattened = "\n".join(
                        f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
                    )
                    generation_config = {
                        "max_output_tokens": max_tokens_value,
                        "temperature": temperature_value if temperature_value is not None else 0,
                    }
                    if top_p_value is not None:
                        generation_config["top_p"] = top_p_value
                    response = self._client.generate_content(  # type: ignore[union-attr]
                        flattened, generation_config=generation_config
                    )
                    text = getattr(response, "text", None)
                    if not text and getattr(response, "candidates", None):
                        candidate = response.candidates[0]
                        text = getattr(candidate, "content", None)
                        if text and isinstance(text, list):
                            text = " ".join(part.text for part in text if getattr(part, "text", None))
                    if not text:
                        raise RuntimeError("Google Generative AI response did not contain text output")
                    return str(text).strip()

                if self._provider == "deepseek":
                    session = self._client["session"]
                    base_url = self._client["base_url"]
                    endpoint = base_url + "/v1/chat/completions"
                    payload = {
                        "model": self._model_name,
                        "messages": messages,
                        "max_tokens": max_tokens_value,
                        "temperature": temperature_value if temperature_value is not None else 0.7,
                    }
                    if top_p_value is not None:
                        payload["top_p"] = top_p_value
                    response = session.post(endpoint, json=payload, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise RuntimeError("DeepSeek response did not contain any choices")
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    reasoning = message.get("reasoning") if isinstance(message, dict) else None
                    if not content:
                        raise RuntimeError("DeepSeek response message did not include content")
                    content_text = str(content).strip()
                    if self._use_reasoning and reasoning:
                        content_text = self._merge_reasoning(content_text, str(reasoning))
                    return content_text

                raise NotImplementedError(f"Unsupported model provider: {self._provider}")
            except (APIStatusError, OpenAIAPIError, GoogleAPIError, requests.RequestException) as exc:
                last_error = exc
                LOGGER.warning("Model invocation failed: %s", exc)
                if attempt < max_retries:
                    time.sleep(retry_delay)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Model invocation failed: %s", exc)
                if attempt < max_retries:
                    time.sleep(retry_delay)

        raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")

    @staticmethod
    def _extract_reasoning(message: Any) -> str:
        if message is None:
            return ""
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            return str(reasoning).strip()
        if isinstance(message, dict):
            reasoning = message.get("reasoning")
            if reasoning:
                return str(reasoning).strip()
        if hasattr(message, "model_dump"):
            try:
                payload = message.model_dump()
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                reasoning = payload.get("reasoning")
                if reasoning:
                    return str(reasoning).strip()
        return ""

    @staticmethod
    def _merge_reasoning(content: str, reasoning: str) -> str:
        if not content or not reasoning:
            return content
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return content
        if isinstance(payload, dict):
            payload["reasoning"] = reasoning
            return json.dumps(payload, ensure_ascii=False)
        return content
