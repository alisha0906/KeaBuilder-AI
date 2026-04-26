"""
KeaBuilder — Similarity Search Service
Task 4: Orchestrates indexing, retrieval, and matching logic

This is the main business logic layer:
  - AssetIndexer:     indexes new assets when uploaded
  - SimilaritySearch: answers search queries (image-to-image, text-to-text,
                      text-to-image cross-modal)
  - Demo seed data:   pre-populates the store so the demo runs immediately
"""

import uuid
import time
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Optional

from embeddings import (
    Asset, EmbeddingEngine, EmbeddingRecord,
    VectorStore, SearchResult, SearchResponse,
    SIMILARITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Asset Indexer
# Called whenever a user uploads an asset to KeaBuilder
# ---------------------------------------------------------------------------

class AssetIndexer:
    """
    Indexes an asset the moment it's uploaded.

    Flow:
      1. Compute content_hash (SHA-256 of raw bytes)
      2. Check for exact duplicates via hash — skip embedding if duplicate
      3. Generate embedding(s) via EmbeddingEngine
      4. Store asset metadata + embedding(s) in VectorStore
    """

    def __init__(self, store: VectorStore, engine: EmbeddingEngine):
        self.store = store
        self.engine = engine
        self._hash_index: dict[str, str] = {}   # content_hash → asset_id

    def index(self, asset: Asset) -> dict:
        """
        Index a new asset. Idempotent — re-indexing the same content is a no-op.
        Returns status and embedding count.
        """
        # --- Deduplication ---
        if asset.content_hash in self._hash_index:
            existing_id = self._hash_index[asset.content_hash]
            return {
                "status": "duplicate",
                "asset_id": asset.asset_id,
                "duplicate_of": existing_id,
                "embeddings_created": 0,
            }

        # --- Store asset metadata ---
        self.store.upsert_asset(asset)
        self._hash_index[asset.content_hash] = asset.asset_id

        # --- Generate and store embeddings ---
        records = self.engine.embed_asset(asset)
        for rec in records:
            self.store.upsert_embedding(rec)

        return {
            "status": "indexed",
            "asset_id": asset.asset_id,
            "embeddings_created": len(records),
            "modalities": [r.modality for r in records],
        }


# ---------------------------------------------------------------------------
# Similarity Search
# ---------------------------------------------------------------------------

class SimilaritySearch:
    """
    Answers three types of query:

    1. Image → similar images
       Use case: "Find templates that look like this design"
       Method:   embed query image with CLIP → search image index

    2. Text → similar text assets
       Use case: "Find templates with similar copy to this headline"
       Method:   embed query text with Sentence-BERT → search text index

    3. Text → similar images (cross-modal)
       Use case: "Find images matching 'minimalist skincare brand'"
       Method:   embed query text with CLIP text encoder → search image index
                 (works because CLIP maps text + images to the same space)
    """

    def __init__(self, store: VectorStore, engine: EmbeddingEngine):
        self.store = store
        self.engine = engine

    def find_similar_to_asset(
        self,
        asset_id: str,
        top_k: int = 5,
        threshold: float = SIMILARITY_THRESHOLD,
        filter_user_id: Optional[str] = None,
    ) -> SearchResponse:
        """Find assets similar to an existing indexed asset."""
        start = int(time.time() * 1000)

        asset = self.store.get_asset(asset_id)
        if not asset:
            return SearchResponse(
                query_asset_id=asset_id,
                query_text=None,
                query_modality="unknown",
                results=[],
                total_found=0,
                search_latency_ms=0,
                model_used="none",
                threshold_used=threshold,
            )

        # Pick modality and embed
        if asset.asset_type in ("image", "template"):
            modality = "image"
            query_vec = self.engine.embed_image(asset).vector
            model = self.engine.IMAGE_MODEL
        else:
            modality = "text"
            query_vec = self.engine.embed_text(asset).vector
            model = self.engine.TEXT_MODEL

        raw = self.store.search(
            query_vector=query_vec,
            modality=modality,
            top_k=top_k,
            threshold=threshold,
            filter_user_id=filter_user_id,
            exclude_asset_id=asset_id,   # never return the query asset itself
        )

        results = self._build_results(raw, f"{modality}_embedding")
        latency = int(time.time() * 1000) - start

        return SearchResponse(
            query_asset_id=asset_id,
            query_text=None,
            query_modality=modality,
            results=results,
            total_found=len(results),
            search_latency_ms=latency,
            model_used=model,
            threshold_used=threshold,
        )

    def find_similar_to_text(
        self,
        query_text: str,
        search_mode: str = "text",    # "text" | "image" (cross-modal)
        top_k: int = 5,
        threshold: float = SIMILARITY_THRESHOLD,
        filter_user_id: Optional[str] = None,
        filter_asset_type: Optional[str] = None,
    ) -> SearchResponse:
        """
        Find assets similar to a text query.

        search_mode="text":  semantic text matching (Sentence-BERT)
        search_mode="image": cross-modal — find IMAGES matching a text description (CLIP)
        """
        start = int(time.time() * 1000)

        if search_mode == "image":
            query_vec = self.engine.embed_query_text(query_text, modality="image")
            model = f"{self.engine.IMAGE_MODEL} (text encoder)"
            modality = "image"
        else:
            query_vec = self.engine.embed_query_text(query_text, modality="text")
            model = self.engine.TEXT_MODEL
            modality = "text"

        raw = self.store.search(
            query_vector=query_vec,
            modality=modality,
            top_k=top_k,
            threshold=threshold,
            filter_user_id=filter_user_id,
            filter_asset_type=filter_asset_type,
        )

        results = self._build_results(raw, f"{modality}_embedding")
        latency = int(time.time() * 1000) - start

        return SearchResponse(
            query_asset_id=None,
            query_text=query_text,
            query_modality=modality,
            results=results,
            total_found=len(results),
            search_latency_ms=latency,
            model_used=model,
            threshold_used=threshold,
        )

    def _build_results(
        self,
        raw: list[tuple[EmbeddingRecord, float]],
        matched_on: str,
    ) -> list[SearchResult]:
        results = []
        for emb, score in raw:
            asset = self.store.get_asset(emb.asset_id)
            if not asset:
                continue
            results.append(SearchResult(
                asset_id=asset.asset_id,
                user_id=asset.user_id,
                asset_type=asset.asset_type,
                storage_url=asset.storage_url,
                text_content=asset.text_content,
                similarity_score=round(score, 4),
                tags=asset.tags,
                matched_on=matched_on,
            ))
        return results


# ---------------------------------------------------------------------------
# Demo seed data + factory
# ---------------------------------------------------------------------------

def _make_asset(
    asset_type: str,
    user_id: str,
    name: str,
    tags: list[str],
    text_content: Optional[str] = None,
    seed: str = "",
) -> Asset:
    content_src = text_content or name + seed
    content_hash = hashlib.sha256(content_src.encode()).hexdigest()
    aid = f"{asset_type[:3]}_{uuid.uuid4().hex[:8]}"
    return Asset(
        asset_id=aid,
        user_id=user_id,
        asset_type=asset_type,
        content_hash=content_hash,
        storage_url=f"https://cdn.keabuilder.com/assets/{user_id}/{aid}.{'png' if asset_type != 'text' else 'txt'}",
        text_content=text_content,
        tags=tags,
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata={"name": name},
    )


DEMO_ASSETS = [
    # Images
    _make_asset("image", "user_001", "skincare hero banner", ["skincare", "hero", "minimalist"], seed="a"),
    _make_asset("image", "user_001", "skincare product flat lay", ["skincare", "product", "marble"], seed="b"),
    _make_asset("image", "user_002", "fitness athlete action shot", ["fitness", "sport", "dynamic"], seed="c"),
    _make_asset("image", "user_002", "fitness app mockup screen", ["fitness", "app", "mobile"], seed="d"),
    _make_asset("image", "user_003", "luxury watch close-up", ["luxury", "watch", "product"], seed="e"),
    _make_asset("image", "user_003", "luxury jewellery lifestyle", ["luxury", "jewellery", "lifestyle"], seed="f"),

    # Templates
    _make_asset("template", "user_001", "skincare landing page", ["skincare", "landing", "funnel"],
                text_content="Glow from within. Our botanically-sourced serum transforms your skin in 14 days.", seed="g"),
    _make_asset("template", "user_002", "fitness lead capture", ["fitness", "lead", "cta"],
                text_content="Start your 30-day transformation. Join 50,000 athletes who changed their lives.", seed="h"),
    _make_asset("template", "user_003", "luxury product showcase", ["luxury", "ecommerce", "premium"],
                text_content="Crafted for those who demand perfection. Limited edition, timeless design.", seed="i"),

    # Text assets
    _make_asset("text", "user_001", "skincare CTA copy", ["skincare", "cta"],
                text_content="Reveal your best skin. Try free for 30 days — no credit card required."),
    _make_asset("text", "user_002", "fitness headline", ["fitness", "headline"],
                text_content="Stop dreaming about the body you want. Start building it today."),
    _make_asset("text", "user_003", "luxury tagline", ["luxury", "brand"],
                text_content="Where craftsmanship meets elegance. Since 1947."),
    _make_asset("text", "user_001", "skincare email subject", ["skincare", "email"],
                text_content="Your skin deserves better. Here's what's possible in 2 weeks."),
]


def build_demo_system() -> tuple[AssetIndexer, SimilaritySearch, VectorStore]:
    """Bootstrap a fully populated demo system."""
    store = VectorStore()
    engine = EmbeddingEngine()
    indexer = AssetIndexer(store, engine)
    search = SimilaritySearch(store, engine)

    print(f"Indexing {len(DEMO_ASSETS)} demo assets...")
    for asset in DEMO_ASSETS:
        indexer.index(asset)
    print(f"Index ready. Stats: {json.dumps(store.stats(), indent=2)}\n")

    return indexer, search, store


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    indexer, search, store = build_demo_system()

    # --- Demo 1: Image → similar images ---
    print("=" * 60)
    print("Demo 1: Image → similar images")
    print("Query: skincare hero banner")
    print("=" * 60)
    query_asset = DEMO_ASSETS[0]
    resp = search.find_similar_to_asset(query_asset.asset_id, top_k=3, threshold=0.0)
    print(json.dumps(asdict(resp), indent=2))

    # --- Demo 2: Text query → similar text assets ---
    print("\n" + "=" * 60)
    print("Demo 2: Text → similar text assets")
    print('Query: "free trial skincare offer"')
    print("=" * 60)
    resp2 = search.find_similar_to_text(
        "free trial skincare offer", search_mode="text", top_k=3, threshold=0.0
    )
    print(json.dumps(asdict(resp2), indent=2))

    # --- Demo 3: Text → images (cross-modal CLIP search) ---
    print("\n" + "=" * 60)
    print("Demo 3: Text → images (cross-modal)")
    print('Query: "minimalist skincare product photography"')
    print("=" * 60)
    resp3 = search.find_similar_to_text(
        "minimalist skincare product photography",
        search_mode="image",
        top_k=3,
        threshold=0.0,
    )
    print(json.dumps(asdict(resp3), indent=2))

    # --- Demo 4: Duplicate detection ---
    print("\n" + "=" * 60)
    print("Demo 4: Duplicate asset detection")
    print("=" * 60)
    duplicate = _make_asset(
        "image", "user_001", "skincare hero banner",
        ["skincare", "hero", "minimalist"], seed="a"
    )
    result = indexer.index(duplicate)
    print(json.dumps(result, indent=2))