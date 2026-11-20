

import hashlib
import math
from typing import List, Optional
from app.config import settings


class EmbeddingsService:
    def __init__(
        self,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        self.backend = (backend or settings.embedding_backend).lower()
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.embedding_model
        self.dim = dim or settings.embedding_dim

        self._client = None
        if self.backend == "gemini":
            self._init_gemini()
        elif self.backend == "mock":
            pass
        else:
            raise ValueError(f"Unsupported EMBEDDING_BACKEND: '{self.backend}'. Must be 'gemini' or 'mock'.")

    def _init_gemini(self):
        if not self.api_key or self.api_key.startswith("your-"):
            raise ValueError(
                "EMBEDDING_BACKEND is set to 'gemini', but GEMINI_API_KEY is missing or invalid. "
                "Set GEMINI_API_KEY in .env or switch EMBEDDING_BACKEND=mock for offline development."
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini embedding client: {e}")

    def _deterministic_mock_vector(self, text: str) -> List[float]:
        """Normalized deterministic pseudo-embedding using SHA256 n-gram word projection."""
        words = text.lower().split()
        vec = [0.0] * self.dim
        for word in words:
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            return [x / norm for x in vec]
        else:
            res = [0.0] * self.dim
            res[0] = 1.0
            return res

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.backend == "mock":
            return [self._deterministic_mock_vector(t) for t in texts]

        # Gemini backend
        try:
            results = []
            for text in texts:
                resp = self._client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                if hasattr(resp, "embeddings") and resp.embeddings:
                    vec = list(resp.embeddings[0].values)
                elif hasattr(resp, "embedding"):
                    vec = list(resp.embedding.values)
                else:
                    vec = list(resp)
                results.append(vec)
            return results
        except Exception as e:
            raise RuntimeError(f"Gemini embedding API call failed: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim

        if self.backend == "mock":
            return self._deterministic_mock_vector(text)

        # Gemini backend
        try:
            resp = self._client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            if hasattr(resp, "embeddings") and resp.embeddings:
                return list(resp.embeddings[0].values)
            elif hasattr(resp, "embedding"):
                return list(resp.embedding.values)
            else:
                return list(resp)
        except Exception as e:
            raise RuntimeError(f"Gemini query embedding failed: {str(e)}")

    def get_metadata(self) -> dict:
        return {
            "backend": self.backend,
            "model_name": self.model_name if self.backend == "gemini" else "mock-sha256",
            "dimensionality": self.dim,
        }
