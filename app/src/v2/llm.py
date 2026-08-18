from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.src.v2.config import settings


class LLMError(RuntimeError):
    pass


class ChatLLMClient:
    def __init__(self) -> None:
        provider = settings.llm_provider.strip().lower()
        if provider == "gemini":
            self.client = GeminiChatClient()
        elif provider == "groq":
            self.client = GroqChatClient()
        else:
            raise LLMError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    @property
    def model(self) -> str:
        return self.client.model

    def is_configured(self) -> bool:
        return self.client.is_configured()

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return self.client.generate_text(system_prompt, user_prompt)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.client.generate_json(system_prompt, user_prompt)


class GroqChatClient:
    def __init__(self) -> None:
        self.model = settings.groq_model
        self.fallback_model = settings.groq_fallback_model

    def is_configured(self) -> bool:
        return bool(settings.groq_api_key)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_configured():
            raise LLMError("GROQ_API_KEY is not configured")
        try:
            from groq import Groq
        except Exception as exc:
            raise LLMError("groq package is not installed") from exc

        client = Groq(api_key=settings.groq_api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            completion = client.chat.completions.create(
                model=self.model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_output_tokens,
                messages=messages,
            )
        except Exception as exc:
            if not self.fallback_model or self.fallback_model == self.model:
                raise LLMError(str(exc)) from exc
            completion = client.chat.completions.create(
                model=self.fallback_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_output_tokens,
                messages=messages,
            )
        return _strip_thinking(completion.choices[0].message.content or "")

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        text = self.generate_text(system_prompt, user_prompt)
        try:
            return json.loads(_extract_json_object(text))
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM did not return valid JSON: {text[:200]}") from exc


class GeminiChatClient:
    def __init__(self) -> None:
        self.model = settings.gemini_model

    def is_configured(self) -> bool:
        return bool(settings.gemini_api_key)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_configured():
            raise LLMError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": settings.llm_max_output_tokens,
            },
        }
        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": settings.gemini_api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(str(part.get("text") or "") for part in parts)
        except Exception as exc:
            raise LLMError(f"Gemini response did not contain text: {str(data)[:200]}") from exc
        return _strip_thinking(text)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        text = self.generate_text(system_prompt, user_prompt)
        try:
            return json.loads(_extract_json_object(text))
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM did not return valid JSON: {text[:200]}") from exc


GroqLlamaClient = GroqChatClient


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    start = text.lower().find("<think>")
    if start >= 0:
        marker = "</think>"
        end = text.lower().find(marker, start)
        if end >= 0:
            text = (text[:start] + text[end + len(marker):]).strip()
        else:
            text = text[:start].strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    return text


def _extract_json_object(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text
