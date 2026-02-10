import os
import json
import asyncio
from typing import Any, Dict, Optional

import httpx


class GroqRateLimitError(Exception):
    """Raised when Groq returns 429 Too Many Requests."""
    pass


class GroqHTTP:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def chat_json(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: float = 30.0,
        retries: int = 2,
        backoff_s: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Calls Groq Chat Completions and expects JSON output.
        - Retries on 429 and transient 5xx / network errors.
        - Raises GroqRateLimitError on final 429.
        """
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            # If your prompts enforce JSON strictly, this helps some providers:
            "response_format": {"type": "json_object"},
        }

        last_err: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    r = await client.post(self.base_url, headers=headers, json=payload)

                if r.status_code == 429:
                    # retry with backoff; if last attempt -> raise special error
                    if attempt >= retries:
                        raise GroqRateLimitError(r.text)
                    await asyncio.sleep(backoff_s * (2 ** attempt))
                    continue

                # retry on transient server errors
                if 500 <= r.status_code < 600:
                    if attempt >= retries:
                        r.raise_for_status()
                    await asyncio.sleep(backoff_s * (2 ** attempt))
                    continue

                r.raise_for_status()

                data = r.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON strictly
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If model returned non-json, wrap it
                    return {"raw": content}

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                last_err = e
                if attempt >= retries:
                    raise
                await asyncio.sleep(backoff_s * (2 ** attempt))
            except Exception as e:
                last_err = e
                raise

        # should not happen
        raise last_err or RuntimeError("Unknown Groq error")
