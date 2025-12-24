# sample/embedding_provider.py
from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    def __init__(self):
        self.openai = OpenAI()
        self.sroberta = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def embed(self, text: str, model: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []

        if model == "openai":
            resp = self.openai.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return resp.data[0].embedding

        elif model == "sroberta":
            return self.sroberta.encode(text).tolist()

        else:
            raise ValueError(f"Unsupported embedding model: {model}")
