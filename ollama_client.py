"""
عميل Ollama - الاتصال بـ Ollama API
يدعم: list models, generate, chat, pull, delete
"""
import httpx
import json
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any
from backend.config import settings
from backend.schemas import ChatMessage

class OllamaClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip('/')
        self.client = httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT)
        self._available = None
    
    async def is_available(self) -> bool:
        """فحص توفر Ollama"""
        if self._available is not None:
            return self._available
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            self._available = response.status_code == 200
            return self._available
        except Exception:
            self._available = False
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """قائمة النماذج المثبتة"""
        if not await self.is_available():
            return []
        response = await self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return data.get('models', [])
    
    async def show_model(self, model_name: str) -> Dict[str, Any]:
        """معلومات تفصيلية عن نموذج"""
        response = await self.client.post(
            f"{self.base_url}/api/show",
            json={"name": model_name}
        )
        response.raise_for_status()
        return response.json()
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """توليد نص (non-streaming)"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system:
            payload["system"] = system
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json().get('response', '')

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        """توليد نص (streaming فقط)"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system:
            payload["system"] = system

        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """محادثة (non-streaming)"""
        payload = {
            "model": model,
            "messages": [m.dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return {
            'content': result.get('message', {}).get('content', ''),
            'usage': result.get('eval_count', 0)
        }

    async def chat_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        """محادثة (streaming فقط)"""
        payload = {
            "model": model,
            "messages": [m.dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data and 'content' in data['message']:
                            yield data['message']['content']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def pull_model(self, model_name: str) -> AsyncGenerator[str, None]:
        """تحميل نموذج من Ollama registry"""
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": model_name, "stream": True}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get('status', '')
                    except:
                        continue
    
    async def delete_model(self, model_name: str) -> bool:
        """حذف نموذج"""
        response = await self.client.delete(
            f"{self.base_url}/api/delete",
            json={"name": model_name}
        )
        return response.status_code == 200
    
    async def embed(self, model: str, text: str) -> List[float]:
        """توليد embeddings"""
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text}
        )
        response.raise_for_status()
        return response.json().get('embedding', [])
    
    async def close(self):
        await self.client.aclose()

# Instance واحد
ollama_client = OllamaClient()