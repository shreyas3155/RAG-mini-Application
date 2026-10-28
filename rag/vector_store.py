"""
Vector Store Module
Clean, fast, persistent Vector Database built with NumPy cosine similarity
and Google Gemini Embeddings (`gemini-embedding-001`).

Industry Value for Interviews:
Demonstrates direct mastery of vector math, embedding indexing,
and top-k cosine similarity search:
    Cosine Similarity = (A . B) / (||A|| * ||B||)
"""

import os
import json
from typing import List, Dict, Any, Optional
import time
import numpy as np
import requests


def get_gemini_embedding(text: str, api_key: str, model: str = "text-embedding-004") -> List[float]:
    """
    Generates dense vector embeddings using Google's embedding model.
    Includes exponential backoff retry and model fallback for 503/429 resilience.
    """
    candidate_models = [model, "gemini-embedding-001"] if model != "gemini-embedding-001" else ["gemini-embedding-001", "text-embedding-004"]
    last_error = ""

    payload = {
        "content": {"parts": [{"text": text[:2048]}]}
    }

    for current_model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:embedContent?key={api_key}"
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    values = data.get("embedding", {}).get("values", [])
                    if values:
                        return values
                
                last_error = f"{current_model} returned {response.status_code}: {response.text}"
                if response.status_code in (503, 429):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                else:
                    break
            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(1.5)

    raise RuntimeError(f"Embedding API error after retries: {last_error}")


class VectorStoreManager:
    """
    Persistent Vector Store that indexes text chunks and their embeddings,
    saving to local disk (JSON vector store) and performing vector similarity search.
    """

    def __init__(self, persist_directory: str = "./vector_store_data", collection_name: str = "rag_store"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.db_file = os.path.join(persist_directory, f"{collection_name}.json")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D)
        
        self._load()

    def _load(self):
        """Loads vectors and documents from disk if exists."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ids = data.get("ids", [])
                    self.documents = data.get("documents", [])
                    self.metadatas = data.get("metadatas", [])
                    embeds = data.get("embeddings", [])
                    if embeds:
                        self.embeddings = np.array(embeds, dtype=np.float32)
            except Exception as e:
                print(f"Error loading vector store: {e}")

    def _save(self):
        """Persists vectors and metadata to disk."""
        data = {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else []
        }
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def add_chunks(self, chunks: List[Dict[str, Any]], api_key: str) -> int:
        """
        Embeds each chunk and adds it to the vector store.
        """
        if not chunks:
            return 0

        new_ids = []
        new_docs = []
        new_metas = []
        new_embeds = []

        for chunk in chunks:
            chunk_id = chunk["id"]
            if chunk_id in self.ids:
                continue  # Avoid duplicate chunk

            text = chunk["text"]
            embedding = get_gemini_embedding(text, api_key=api_key)
            if not embedding:
                continue

            new_ids.append(chunk_id)
            new_docs.append(text)
            new_metas.append(chunk["metadata"])
            new_embeds.append(embedding)

        if not new_ids:
            return 0

        self.ids.extend(new_ids)
        self.documents.extend(new_docs)
        self.metadatas.extend(new_metas)

        new_embed_matrix = np.array(new_embeds, dtype=np.float32)
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embed_matrix
        else:
            self.embeddings = np.vstack([self.embeddings, new_embed_matrix])

        self._save()
        return len(new_ids)

    def query_similar_chunks(
        self,
        query_text: str,
        api_key: str,
        n_results: int = 3,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity between query embedding and all indexed chunk vectors:
        Cosine Similarity = (q . d) / (||q|| * ||d||)
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # 1. Embed query
        query_vec = np.array(get_gemini_embedding(query_text, api_key=api_key), dtype=np.float32)
        
        # 2. Compute Cosine Similarities across all chunks (Vectorized Matrix Operation)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        # Avoid division by zero
        doc_norms[doc_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10

        similarities = np.dot(self.embeddings, query_vec) / (doc_norms * query_norm)

        # 3. Filter by source if requested
        valid_indices = []
        for idx in range(len(self.documents)):
            if source_filter:
                if self.metadatas[idx].get("source") == source_filter:
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        # 4. Sort by highest similarity
        valid_indices.sort(key=lambda i: similarities[i], reverse=True)
        top_indices = valid_indices[:n_results]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(similarities[idx]),
                "distance": float(1.0 - similarities[idx])
            })

        return results

    def count(self) -> int:
        return len(self.documents)

    def clear(self):
        """Resets the vector database."""
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = None
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
