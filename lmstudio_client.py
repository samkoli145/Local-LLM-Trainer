"""
عميل LM Studio - OpenAI-compatible API
يستخدم كـ fallback عند عدم توفر Ollama
"""
import httpx
from typing import List, Optional, Dict, Any, AsyncGenerator
from backend.config import settings
from backend.schemas import ChatMessage

class LMStudioClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or settings.LM_STUDIO_BASE_URL).rstrip('/')
        self.client = httpx.AsyncClient(timeout=300)
        self._available = None
    
    async def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            response = await self.client.get(f"{self.base_url}/models")
            self._available = response.status_code == 200
            return self._available
        except:
            self._available = False
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        if not await self.is_available():
            return []
        response = await self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json().get('data', [])
    
    async def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """محادثة عبر OpenAI-compatible API"""
        payload = {
            "model": model,
            "messages": [m.dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return {
            'content': data['choices'][0]['message']['content'],
            'usage': data.get('usage', {})
        }

    async def chat_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        """محادثة (streaming)"""
        payload = {
            "model": model,
            "messages": [m.dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            yield delta['content']
                    except:
                        continue
    
    async def close(self):
        await self.client.aclose()

lmstudio_client = LMStudioClient()