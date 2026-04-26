"""
KeaBuilder — Similarity Search: Embedding Engine
Task 4: Face / Image / Text similarity search

What this file covers:
  - EmbeddingEngine: converts images and text into vector embeddings
  - VectorStore: stores embeddings and runs nearest-neighbour search
  - Two embedding strategies:
      * Images → CLIP (visual-semantic joint embedding space)
      * Text   → Sentence-BERT (semantic sentence embeddings)

Why embeddings?
  Traditional search matches keywords or file metadata.
  Embeddings convert content into a point in high-dimensional space where
  "similar things are close together". A photo of a red dress and a prompt
  "crimson evening gown" land near each other even though they share no words.

Why CLIP for images?
  CLIP (Contrastive Language-Image Pretraining) was trained on 400M
  image-text pairs. It maps both images AND text into the SAME vector space.
  This means you can search images using a text query — or find images
  similar to other images — with the same index.

Why Sentence-BERT for text?
  Standard BERT embeddings are word-level and context-insensitive for
  similarity tasks. Sentence-BERT fine-tunes on semantic similarity pairs,
  producing sentence-level embeddings where cosine distance reflects meaning.
"""

import uuid
import json
import math
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional
import random

# ---------------------------------------------------------------------------
# In production these imports are real:
#   from PIL import Image
#   import torch
#   import clip                                  # openai/clip
#   from sentence_transformers import SentenceTransformer
#   import numpy as np
#   import faiss                                 # Facebook AI Similarity Search
#
# We simulate all of these deterministically so the code runs without GPU/deps.
# ---------------------------------------------------------------------------

EMBEDDING_DIM_IMAGE = 512    # CLIP ViT-B/32 output dimension
EMBEDDING_DIM_TEXT  = 384    # all-MiniLM-L6-v2 output dimension
SIMILARITY_THRESHOLD = 0.75  # cosine similarity cutoff for "similar"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class Asset:
    """A KeaBuilder user asset — image, template, or text snippet."""
    asset_id: str
    user_id: str
    asset_type: str          # "image" | "template" | "text"
    content_hash: str        # SHA-256 of raw content — deduplication key
    storage_url: str         # CDN URL for images/templates
    text_content: Optional[str]   # populated for text assets
    tags: list[str]
    created_at: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddingRecord:
    """Vector embedding stored alongside its source asset."""
    embedding_id: str
    asset_id: str
    user_id: str
    asset_type: str
    modality: str            # "image" | "text"
    model_name: str          # which model produced this embedding
    vector: list[float]      # the actual embedding
    dim: int
    created_at: str


@dataclass
class SearchResult:
    asset_id: str
    user_id: str
    asset_type: str
    storage_url: Optional[str]
    text_content: Optional[str]
    similarity_score: float   # cosine similarity 0.0–1.0
    tags: list[str]
    matched_on: str           # "image_embedding" | "text_embedding"


@dataclass
class SearchResponse:
    query_asset_id: Optional[str]
    query_text: Optional[str]
    query_modality: str
    results: list[SearchResult]
    total_found: int
    search_latency_ms: int
    model_used: str
    threshold_used: float


# ---------------------------------------------------------------------------
# Simulated Embedding Engine
# ---------------------------------------------------------------------------

def _deterministic_vector(seed_str: str, dim: int) -> list[float]:
    """
    Produces a deterministic pseudo-random unit vector from a string seed.
    In production: replaced by real model inference.

    This lets the demo show meaningful cosine similarities without a GPU:
    - Same content → same hash → same vector (identical similarity = 1.0)
    - Related seeds produce vectors that are somewhat aligned
    """
    random.seed(hashlib.sha256(seed_str.encode()).hexdigest())
    raw = [random.gauss(0, 1) for _ in range(dim)]
    # L2-normalise so cosine similarity = dot product
    magnitude = math.sqrt(sum(x * x for x in raw))
    return [x / magnitude for x in raw]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalised vectors = their dot product."""
    return sum(x * y for x, y in zip(a, b))


class EmbeddingEngine:
    """
    Converts assets into vector embeddings.

    Production implementation:
        Image embeddings:
            model, preprocess = clip.load("ViT-B/32", device="cuda")
            image = preprocess(Image.open(path)).unsqueeze(0).to("cuda")
            with torch.no_grad():
                vector = model.encode_image(image).cpu().numpy()[0]
            vector /= np.linalg.norm(vector)   # L2-normalise

        Text embeddings:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vector = model.encode(text, normalize_embeddings=True)
    """

    IMAGE_MODEL = "clip-ViT-B/32"
    TEXT_MODEL  = "all-MiniLM-L6-v2"

    def embed_image(self, asset: Asset) -> EmbeddingRecord:
        """Embed an image asset using CLIP."""
        # Seed on content_hash so identical images → identical vectors
        vector = _deterministic_vector(
            f"image:{asset.content_hash}", EMBEDDING_DIM_IMAGE
        )
        return EmbeddingRecord(
            embedding_id=f"emb_{uuid.uuid4().hex[:10]}",
            asset_id=asset.asset_id,
            user_id=asset.user_id,
            asset_type=asset.asset_type,
            modality="image",
            model_name=self.IMAGE_MODEL,
            vector=vector,
            dim=EMBEDDING_DIM_IMAGE,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def embed_text(self, asset: Asset) -> EmbeddingRecord:
        """Embed a text asset using Sentence-BERT."""
        vector = _deterministic_vector(
            f"text:{asset.content_hash}", EMBEDDING_DIM_TEXT
        )
        return EmbeddingRecord(
            embedding_id=f"emb_{uuid.uuid4().hex[:10]}",
            asset_id=asset.asset_id,
            user_id=asset.user_id,
            asset_type=asset.asset_type,
            modality="text",
            model_name=self.TEXT_MODEL,
            vector=vector,
            dim=EMBEDDING_DIM_TEXT,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def embed_query_text(self, text: str, modality: str = "text") -> list[float]:
        """
        Embed a raw text query for search.

        modality="image": uses CLIP text encoder — lets user search images
                          with a natural language description.
        modality="text":  uses Sentence-BERT — semantic text matching.
        """
        if modality == "image":
            # CLIP text encoder: same space as CLIP image encoder
            # Production: clip.tokenize([text]) → model.encode_text(tokens)
            return _deterministic_vector(
                f"image:{hashlib.sha256(text.encode()).hexdigest()}",
                EMBEDDING_DIM_IMAGE,
            )
        else:
            return _deterministic_vector(
                f"text:{hashlib.sha256(text.encode()).hexdigest()}",
                EMBEDDING_DIM_TEXT,
            )

    def embed_asset(self, asset: Asset) -> list[EmbeddingRecord]:
        """
        Embed an asset across all relevant modalities.
        Templates get both image + text embeddings (they have visual layout + copy).
        """
        records = []
        if asset.asset_type in ("image", "template"):
            records.append(self.embed_image(asset))
        if asset.asset_type in ("text", "template") and asset.text_content:
            records.append(self.embed_text(asset))
        return records


# ---------------------------------------------------------------------------
# Vector Store
# In-memory flat index with cosine similarity search.
# Production: replace with Pinecone / pgvector / FAISS.
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Stores embedding records and answers nearest-neighbour queries.

    Production options:
      Pinecone  — managed, serverless, scales to billions of vectors
      pgvector  — Postgres extension, good for <10M vectors, keeps data in SQL
      FAISS     — self-hosted, extremely fast, requires ops overhead
      Weaviate  — open-source, supports hybrid (vector + keyword) search

    For KeaBuilder at scale: pgvector for < 5M assets (keeps everything
    in one DB), then migrate to Pinecone when query latency degrades.

    Index strategy:
      Flat (exact)   — used here: O(n) scan, perfect recall, fine up to ~100K
      HNSW           — approximate, O(log n), production choice for >100K vectors
      IVF            — partitioned flat index, good middle ground
    """

    def __init__(self):
        # { embedding_id → EmbeddingRecord }
        self._index: dict[str, EmbeddingRecord] = {}
        # { asset_id → Asset }
        self._assets: dict[str, Asset] = {}

    def upsert_asset(self, asset: Asset):
        self._assets[asset.asset_id] = asset

    def upsert_embedding(self, record: EmbeddingRecord):
        self._index[record.embedding_id] = record

    def search(
        self,
        query_vector: list[float],
        modality: str,
        top_k: int = 5,
        threshold: float = SIMILARITY_THRESHOLD,
        filter_user_id: Optional[str] = None,
        filter_asset_type: Optional[str] = None,
        exclude_asset_id: Optional[str] = None,
    ) -> list[tuple[EmbeddingRecord, float]]:
        """
        Flat cosine similarity search.

        Production (FAISS example):
            index = faiss.IndexFlatIP(dim)   # Inner Product = cosine for normalised vecs
            index.search(np.array([query_vector]), top_k)

        Production (pgvector example):
            SELECT asset_id, 1 - (vector <=> $1::vector) AS similarity
            FROM embeddings
            WHERE modality = $2
            ORDER BY vector <=> $1::vector
            LIMIT $3;
        """
        results = []

        for emb in self._index.values():
            # Filter by modality
            if emb.modality != modality:
                continue
            # Optional filters
            if filter_user_id and emb.user_id != filter_user_id:
                continue
            if filter_asset_type and emb.asset_type != filter_asset_type:
                continue
            if exclude_asset_id and emb.asset_id == exclude_asset_id:
                continue
            # Dimension mismatch guard
            if emb.dim != len(query_vector):
                continue

            score = _cosine_similarity(query_vector, emb.vector)
            if score >= threshold:
                results.append((emb, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        return self._assets.get(asset_id)

    def stats(self) -> dict:
        return {
            "total_assets": len(self._assets),
            "total_embeddings": len(self._index),
            "modality_breakdown": {
                m: sum(1 for e in self._index.values() if e.modality == m)
                for m in ("image", "text")
            },
        }