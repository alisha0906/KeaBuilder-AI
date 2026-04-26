"""
KeaBuilder — AI Content Routing Engine
Task 2: Multi-Provider Content Generation System

Handles routing, fallback, versioning, and asset management
for image, video, and voice generation.
"""

import uuid
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Provider Definitions
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str
    supports: list[str]       # content types this provider handles
    tier: str                 # "free" | "premium"
    cost_per_call: float      # USD estimate
    avg_latency_ms: int       # expected latency


PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    # Image providers
    "stability_ai": ProviderConfig(
        name="stability_ai",
        supports=["image"],
        tier="premium",
        cost_per_call=0.04,
        avg_latency_ms=3000,
    ),
    "dalle": ProviderConfig(
        name="dalle",
        supports=["image"],
        tier="free",
        cost_per_call=0.02,
        avg_latency_ms=4000,
    ),

    # Video providers
    "runway_ml": ProviderConfig(
        name="runway_ml",
        supports=["video"],
        tier="premium",
        cost_per_call=0.50,
        avg_latency_ms=30000,
    ),
    "pika_labs": ProviderConfig(
        name="pika_labs",
        supports=["video"],
        tier="free",
        cost_per_call=0.20,
        avg_latency_ms=45000,
    ),

    # Voice providers
    "elevenlabs": ProviderConfig(
        name="elevenlabs",
        supports=["voice"],
        tier="premium",
        cost_per_call=0.03,
        avg_latency_ms=1500,
    ),
    "openai_tts": ProviderConfig(
        name="openai_tts",
        supports=["voice"],
        tier="free",
        cost_per_call=0.01,
        avg_latency_ms=2000,
    ),
}


# ---------------------------------------------------------------------------
# Routing Table
# Primary + fallback per (content_type, user_plan) combination
# ---------------------------------------------------------------------------

ROUTING_TABLE: dict[tuple[str, str], list[str]] = {
    ("image", "premium"):  ["stability_ai", "dalle"],
    ("image", "free"):     ["dalle", "stability_ai"],
    ("video", "premium"):  ["runway_ml", "pika_labs"],
    ("video", "free"):     ["pika_labs", "runway_ml"],
    ("voice", "premium"):  ["elevenlabs", "openai_tts"],
    ("voice", "free"):     ["openai_tts", "elevenlabs"],
}


# ---------------------------------------------------------------------------
# Provider Base + Concrete Implementations
# ---------------------------------------------------------------------------

class ContentProvider:
    """Abstract base for all content providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    def generate(self, prompt: str, options: dict) -> dict:
        """
        In production: calls the real provider API.
        Here: returns a simulated success response.
        """
        # Simulate network latency (not blocking in real async impl)
        time.sleep(0)

        content_type = self.config.supports[0]
        ext = {"image": "png", "video": "mp4", "voice": "mp3"}[content_type]
        asset_key = f"{content_type}_{uuid.uuid4().hex[:8]}"

        return {
            "raw_url": f"https://provider-cdn.{self.config.name}.io/{asset_key}.{ext}",
            "provider": self.config.name,
            "latency_ms": self.config.avg_latency_ms,
            "cost_usd": self.config.cost_per_call,
            "meta": {
                "prompt": prompt,
                "options": options,
            },
        }

    def health_check(self) -> bool:
        """
        In production: pings the provider's status endpoint.
        Simulates occasional failure for demo purposes.
        """
        # Simulate providers always healthy in demo
        return True


class StabilityAIProvider(ContentProvider):
    pass


class DalleProvider(ContentProvider):
    pass


class RunwayMLProvider(ContentProvider):
    pass


class PikaLabsProvider(ContentProvider):
    pass


class ElevenLabsProvider(ContentProvider):
    pass


class OpenAITTSProvider(ContentProvider):
    pass


PROVIDER_CLASSES = {
    "stability_ai": StabilityAIProvider,
    "dalle": DalleProvider,
    "runway_ml": RunwayMLProvider,
    "pika_labs": PikaLabsProvider,
    "elevenlabs": ElevenLabsProvider,
    "openai_tts": OpenAITTSProvider,
}


# ---------------------------------------------------------------------------
# Asset Store (in-memory for demo; replace with DB in production)
# ---------------------------------------------------------------------------

_asset_store: dict[str, list[dict]] = {}  # asset_id → list of versions


def store_asset(asset_id: str, version: int, payload: dict) -> dict:
    """
    Persists a generated asset. Supports versioning:
    each regeneration appends a new version instead of overwriting.
    """
    record = {
        "asset_id": asset_id,
        "version": version,
        "provider": payload["provider"],
        "content_type": payload["content_type"],
        "asset_url": f"https://cdn.keabuilder.com/assets/{asset_id}/v{version}/{payload['content_type']}_{asset_id}.{payload['ext']}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": payload["prompt"],
        "options": payload.get("options", {}),
        "cost_usd": payload.get("cost_usd", 0),
        "used_in": [],
        "generation_metadata": payload.get("meta", {}),
    }

    if asset_id not in _asset_store:
        _asset_store[asset_id] = []
    _asset_store[asset_id].append(record)

    return record


def get_asset_versions(asset_id: str) -> list[dict]:
    return _asset_store.get(asset_id, [])


# ---------------------------------------------------------------------------
# Routing Engine
# ---------------------------------------------------------------------------

class RoutingError(Exception):
    pass


class RoutingEngine:
    """
    Selects the best provider for a request using the routing table.
    Automatically tries fallback providers on failure.
    """

    def route(
        self,
        content_type: str,
        user_plan: str,
        prompt: str,
        options: dict,
        existing_asset_id: Optional[str] = None,
    ) -> dict:

        # --- Validate content type ---
        if content_type not in ("image", "video", "voice"):
            raise RoutingError(f"Unsupported content type: {content_type}")

        # --- Determine provider chain ---
        route_key = (content_type, user_plan)
        provider_chain = ROUTING_TABLE.get(route_key)
        if not provider_chain:
            raise RoutingError(f"No route defined for {route_key}")

        # --- Try providers in order (primary → fallbacks) ---
        last_error = None
        provider_used = None
        raw_result = None

        for provider_name in provider_chain:
            config = PROVIDER_REGISTRY[provider_name]
            provider_cls = PROVIDER_CLASSES[provider_name]
            provider = provider_cls(config)

            if not provider.health_check():
                last_error = f"{provider_name} failed health check"
                continue

            try:
                raw_result = provider.generate(prompt, options)
                provider_used = provider_name
                break
            except Exception as exc:
                last_error = str(exc)
                continue

        if raw_result is None:
            raise RoutingError(
                f"All providers exhausted for {content_type}. Last error: {last_error}"
            )

        # --- Determine asset ID and version ---
        if existing_asset_id and existing_asset_id in _asset_store:
            asset_id = existing_asset_id
            version = len(_asset_store[asset_id]) + 1
        else:
            asset_id = f"{content_type[:3]}_{uuid.uuid4().hex[:8]}"
            version = 1

        ext_map = {"image": "png", "video": "mp4", "voice": "mp3"}
        ext = ext_map[content_type]

        # --- Store asset ---
        stored = store_asset(
            asset_id,
            version,
            {
                "content_type": content_type,
                "ext": ext,
                "prompt": prompt,
                "options": options,
                "provider": provider_used,
                "cost_usd": raw_result["cost_usd"],
                "meta": raw_result.get("meta", {}),
            },
        )

        # --- Build response ---
        was_fallback = provider_used != provider_chain[0]

        return {
            "asset_id": stored["asset_id"],
            "version": stored["version"],
            "provider": stored["provider"],
            "fallback_used": was_fallback,
            "status": "success",
            "asset_url": stored["asset_url"],
            "created_at": stored["created_at"],
            "cost_usd": stored["cost_usd"],
            "used_in": [],
        }


# ---------------------------------------------------------------------------
# Public API: generate_content()
# Called by app.py / API Gateway
# ---------------------------------------------------------------------------

_routing_engine = RoutingEngine()


def generate_content(payload: dict) -> dict:
    """
    Main entry point.

    Expected payload keys:
        content_type   : "image" | "video" | "voice"
        prompt         : str
        user_plan      : "free" | "premium"
        style          : str (optional)
        duration       : str (optional, video/voice)
        quality        : str (optional)
        asset_id       : str (optional, for regeneration/versioning)
    """
    content_type = payload.get("content_type", "").lower()
    prompt = payload.get("prompt", "")
    user_plan = payload.get("user_plan", "free").lower()

    if not prompt:
        return {"error": "Prompt is required", "status": "failed"}

    options = {
        k: v for k, v in payload.items()
        if k not in ("content_type", "prompt", "user_plan", "asset_id")
    }
    existing_asset_id = payload.get("asset_id")

    try:
        result = _routing_engine.route(
            content_type=content_type,
            user_plan=user_plan,
            prompt=prompt,
            options=options,
            existing_asset_id=existing_asset_id,
        )
        return result
    except RoutingError as exc:
        return {"error": str(exc), "status": "failed"}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    requests = [
        # Premium video — primary provider (Runway ML)
        {
            "content_type": "video",
            "prompt": "Create a product demo video for a skincare brand",
            "style": "professional",
            "duration": "30 sec",
            "user_plan": "premium",
        },
        # Free image — primary provider (DALL·E)
        {
            "content_type": "image",
            "prompt": "Hero banner for a fitness app, bold colours",
            "style": "vibrant",
            "user_plan": "free",
        },
        # Premium voice
        {
            "content_type": "voice",
            "prompt": "Welcome to KeaBuilder. Build funnels that convert.",
            "duration": "15 sec",
            "user_plan": "premium",
        },
    ]

    for req in requests:
        print(f"\n--- {req['content_type'].upper()} | {req['user_plan']} ---")
        result = generate_content(req)
        print(json.dumps(result, indent=2))