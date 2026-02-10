import os
import json
import asyncio
from typing import Any, Dict, Tuple, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv(".env")


class LLMRateLimitError(Exception):
    pass


class LLMHTTP:
    """
    OpenAI-compatible provider (Groq/OpenAI)
    Accepts either:
      - base_url = https://api.openai.com/v1
      - OR base_url = https://api.openai.com/v1/chat/completions
    """
    def __init__(self, api_key: str, base_url: str, model: str, name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name

    def _chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def chat_json(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: float = 30.0,
        retries: int = 1,
        backoff_s: float = 0.8,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }

        url = self._chat_url()
        last_text = None

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    r = await client.post(url, headers=headers, json=payload)
            except httpx.ReadTimeout:
                if attempt >= retries:
                    raise
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            last_text = r.text

            if r.status_code == 429:
                if attempt >= retries:
                    raise LLMRateLimitError(f"{self.name} 429: {r.text}")
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            if 500 <= r.status_code < 600:
                if attempt >= retries:
                    r.raise_for_status()
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}

        raise RuntimeError(f"{self.name} call failed. last={last_text}")



class OllamaHTTP:
    """
    Native Ollama API:
      POST http://127.0.0.1:11434/api/chat
    """

    def __init__(self, base_url: str, model: str, name: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name

    def _ollama_options(self, temperature: float, max_tokens: int) -> Dict[str, Any]:
        # lighter defaults
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX") or 2048)
        max_predict_env = int(os.getenv("OLLAMA_MAX_PREDICT") or 384)

        num_predict = min(int(max_tokens), max_predict_env)

        return {
            "temperature": float(temperature),
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }

    async def chat_json(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 384,
        timeout_s: float = 180.0,   # longer for local
        retries: int = 1,
        backoff_s: float = 1.0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "options": self._ollama_options(temperature=temperature, max_tokens=max_tokens),
        }

        last_text = None

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    r = await client.post(url, json=payload)
            except httpx.ReadTimeout:
                if attempt >= retries:
                    raise
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            last_text = r.text

            if r.status_code == 429:
                if attempt >= retries:
                    raise LLMRateLimitError(f"{self.name} 429: {r.text}")
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            if 500 <= r.status_code < 600:
                if attempt >= retries:
                    r.raise_for_status()
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content") or ""

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}

        raise RuntimeError(f"{self.name} call failed. last={last_text}")


class LLMRouter:
    """
    Provider order:
      1) Groq (if GROQ_API_KEY)
      2) OpenAI (if OPENAI_API_KEY)
      3) Ollama (if OLLAMA_BASE_URL or OLLAMA_MODEL)   <-- last, but works offline

    Returns: (json, provider_name, trace[])
    """

    def __init__(self):
        self.providers: List[Any] = []

        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if groq_key:
            self.providers.append(
                LLMHTTP(
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1/chat/completions",
                    model=(os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
                    name="groq",
                )
            )

        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if openai_key:
            self.providers.append(
                LLMHTTP(
                    api_key=openai_key,
                    base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/chat/completions").strip(),
                    model=(os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
                    name="openai",
                )
            )

        # Ollama config
        ollama_url = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
        ollama_model = (os.getenv("OLLAMA_MODEL") or "").strip()
        if ollama_model:
            self.providers.append(OllamaHTTP(base_url=ollama_url, model=ollama_model, name="ollama"))

        if not self.providers:
            raise RuntimeError("No LLM providers configured. Set GROQ_API_KEY and/or OPENAI_API_KEY and/or OLLAMA_MODEL.")

    def providers_status(self) -> Dict[str, Any]:
        return {
            "configured": [p.name for p in self.providers],
            "models": {p.name: p.model for p in self.providers},
        }

    async def chat_json(self, *args, **kwargs) -> Tuple[Dict[str, Any], str, List[str]]:
        trace: List[str] = []

        for p in self.providers:
            try:
                trace.append(f"try:{p.name}:{p.model}")
                out = await p.chat_json(*args, **kwargs)
                trace.append(f"ok:{p.name}")
                return out, p.name, trace
            except Exception as e:
                msg = str(e)
                if len(msg) > 200:
                    msg = msg[:200] + "…"
                trace.append(f"fail:{p.name}:{type(e).__name__}:{msg}")
                continue

        raise RuntimeError("All LLM providers failed. trace=" + " | ".join(trace))
