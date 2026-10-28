"""
RAG Chain Module
Orchestrates:
1. Question embedding & similarity retrieval from VectorStore
2. Prompt engineering with strict context grounding to minimize hallucinations
3. LLM generation using Google Gemini REST API (Gemini 3 Flash / 3.1 Flash-Lite / 3.8 Flash with auto-failover)
"""

from typing import List, Dict, Any, Optional
import os
import time
import requests
from pathlib import Path


def load_env_file():
    """Reads .env file without external dependencies."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")


load_env_file()

SYSTEM_PROMPT = """You are a helpful, precise AI Research Assistant.
You answer user questions strictly based on the provided retrieved context from uploaded PDF documents.

Guidelines:
1. Truthfulness & Grounding: Base your answer ONLY on the given context. If the context does not contain enough information to answer the question, clearly state: "I couldn't find enough information in the uploaded document to answer that question." Do NOT fabricate facts.
2. Citations: Reference the source document and page number whenever making specific factual claims (e.g., "[Doc: notes.pdf, Page: 3]").
3. Clarity & Structure: Use clear formatting, bullet points, and code snippets when explaining technical topics.
4. Tone: Professional, encouraging, and clear.
"""


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved vector chunks into a clean context block with metadata headers."""
    if not retrieved_chunks:
        return "No relevant context found."

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown Document")
        page = meta.get("page", "?")
        text = chunk.get("text", "").strip()

        block = f"--- [SOURCE {i}: {source} | Page {page}] ---\n{text}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def normalize_model_name(name: str) -> str:
    """Normalizes model names to valid Google API endpoint slugs."""
    clean = name.strip()
    mapping = {
        "gemini 3 flash": "gemini-3-flash",
        "gemini 3.1 flash-lite": "gemini-3.1-flash-lite",
        "gemini 3.8 flash": "gemini-3.8-flash",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini 2.0 flash": "gemini-2.0-flash",
        "gemini 1.5 flash": "gemini-1.5-flash",
        "gemini 1.5 pro": "gemini-1.5-pro",
    }
    return mapping.get(clean.lower(), clean.lower().replace(" ", "-"))


class RAGPipeline:
    """
    End-to-end RAG pipeline coordinating retrieval and generation.
    """

    def __init__(self, vector_store, api_key: Optional[str] = None, model_name: str = "gemini-3-flash"):
        self.vector_store = vector_store
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = normalize_model_name(model_name)

    def set_api_key(self, api_key: str):
        """Update API key at runtime."""
        self.api_key = api_key

    def ask(
        self,
        question: str,
        top_k: int = 3,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs full RAG pipeline for a user question:
        1. Retrieve top-k relevant chunks via vector cosine similarity
        2. Construct grounded prompt
        3. Call Gemini LLM to generate response with sources
        """
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {
                "answer": (
                    "⚠️ **Google Gemini API Key is missing.**\n\n"
                    "Please enter your free Google Gemini API Key in the left sidebar or inside your `.env` file.\n"
                    "Get a key for free from [Google AI Studio](https://aistudio.google.com/app/apikey)."
                ),
                "sources": [],
                "context_used": ""
            }

        # Step 1: Semantic Retrieval
        retrieved_chunks = self.vector_store.query_similar_chunks(
            query_text=question,
            api_key=api_key,
            n_results=top_k,
            source_filter=source_filter
        )

        if not retrieved_chunks:
            return {
                "answer": "No documents have been indexed yet. Please upload a PDF document first in the sidebar!",
                "sources": [],
                "context_used": ""
            }

        context_text = format_context(retrieved_chunks)

        # Step 2: Prompt Engineering
        full_prompt = f"""{SYSTEM_PROMPT}

CONTEXT FROM UPLOADED DOCUMENTS:
{context_text}

USER QUESTION:
{question}

Provide an accurate, well-structured answer with source citations based on the context above:"""

        # Step 3: Call Gemini REST API with Retry and Fallback
        answer_text, error_msg = self._generate_with_retry_and_fallback(full_prompt, api_key)
        if answer_text:
            return {
                "answer": answer_text,
                "sources": retrieved_chunks,
                "context_used": context_text
            }
        else:
            return {
                "answer": error_msg,
                "sources": retrieved_chunks,
                "context_used": context_text
            }

    def _generate_with_retry_and_fallback(self, prompt: str, api_key: str) -> tuple[Optional[str], Optional[str]]:
        """
        Sends generation request with exponential backoff and fallback models
        to handle 503 (high demand) and 429 (rate limit) errors gracefully.
        """
        candidate_models = [self.model_name]
        for fallback in [
            "gemini-3-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]:
            slug = normalize_model_name(fallback)
            if slug not in candidate_models:
                candidate_models.append(slug)

        last_error = ""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024
            }
        }

        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            
            # Retry loop with exponential backoff for transient 503 / 429 errors
            for attempt in range(3):
                try:
                    response = requests.post(url, json=payload, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                return parts[0]["text"], None
                        return "No response text was generated by the model.", None

                    last_error = f"Gemini API ({model}) returned {response.status_code}: {response.text}"
                    
                    # If 503 (High Demand) or 429 (Rate Limit), back off and retry or switch model
                    if response.status_code in (503, 429):
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    else:
                        # Non-transient error (e.g. 400 or 404), move to next candidate model immediately
                        break
                except requests.RequestException as e:
                    last_error = f"Network error with model {model}: {str(e)}"
                    time.sleep(1.5)

        # If all candidates and retries failed
        friendly_error = (
            "⏳ **Google Gemini API is currently experiencing temporary high server demand (503).**\n\n"
            "Google's servers are temporarily busy processing requests across free-tier clusters.\n\n"
            "**What you can do:**\n"
            "- Wait 10–20 seconds and click or ask again.\n"
            "- Try switching to another Gemini model in the sidebar (e.g. `gemini-2.0-flash` or `gemini-1.5-pro`).\n"
            "- If the issue persists, check [Google AI Studio](https://aistudio.google.com/) for service status or create a fresh API key.\n\n"
            f"<details><summary>Technical Error Details</summary><code>{last_error}</code></details>"
        )
        return None, friendly_error

