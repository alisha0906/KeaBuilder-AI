"""
KeaBuilder — Personalised AI Image Generation via LoRA
Task 3: LoRA model integration into the inference pipeline

What this file covers:
  - LoRARegistry: manages user-trained LoRA models (stored as .safetensors)
  - LoRAPipeline: wraps Stable Diffusion + LoRA injection logic
  - InferenceRequest / InferenceResult: typed data contracts
  - generate_personalised_image(): single public entry point called by app.py

LoRA (Low-Rank Adaptation) lets us fine-tune a small set of weights on top
of a frozen base model. For KeaBuilder, each user/brand gets their own LoRA
adapter trained on their uploaded reference images.  At inference time we:
  1. Load the frozen base model (once, shared across all requests)
  2. Inject the user's LoRA weights on top (cheap, ~50ms)
  3. Run the diffusion loop with those weights active
  4. Eject the LoRA weights so the next request starts clean

This means one GPU instance serves all users without reloading the base model.
"""

import os
import uuid
import time
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# In production these would be real imports:
#   from diffusers import StableDiffusionPipeline
#   import torch
# We simulate them here so the code runs without a GPU environment.
# ---------------------------------------------------------------------------

LORA_STORE_PATH = Path(os.getenv("LORA_STORE_PATH", "./lora_weights"))
CDN_BASE = "https://cdn.keabuilder.com/generated"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class LoRAModel:
    """Metadata record for a user's trained LoRA adapter."""
    lora_id: str
    user_id: str
    brand_name: str
    trigger_word: str          # e.g. "acmebrand" — injected into every prompt
    weights_path: str          # path to .safetensors file
    base_model: str            # e.g. "stabilityai/stable-diffusion-xl-base-1.0"
    lora_scale: float          # 0.0–1.0, how strongly LoRA influences output
    training_images: int       # how many reference images were used
    created_at: str
    status: str                # "ready" | "training" | "failed"
    version: int = 1


@dataclass
class InferenceRequest:
    user_id: str
    prompt: str
    lora_id: Optional[str] = None      # None = no personalisation, use base model
    negative_prompt: str = "blurry, low quality, distorted, watermark"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30      # quality vs speed trade-off
    guidance_scale: float = 7.5        # prompt adherence strength
    seed: Optional[int] = None         # for reproducibility
    style_preset: str = "photorealistic"


@dataclass
class InferenceResult:
    request_id: str
    user_id: str
    lora_id: Optional[str]
    trigger_word_injected: bool
    lora_scale_used: float
    asset_url: str
    created_at: str
    latency_ms: int
    seed_used: int
    prompt_final: str          # prompt after trigger word injection
    status: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LoRA Registry
# Manages all trained LoRA adapters per user
# ---------------------------------------------------------------------------

class LoRARegistry:
    """
    In production: backed by PostgreSQL.
    Here: in-memory dict seeded with demo records.
    """

    def __init__(self):
        self._store: dict[str, LoRAModel] = {}
        self._seed_demo_data()

    def _seed_demo_data(self):
        """Pre-load two demo LoRA models so the pipeline can run immediately."""
        demos = [
            LoRAModel(
                lora_id="lora_acme_brand_v1",
                user_id="user_001",
                brand_name="Acme Skincare",
                trigger_word="acmebrand",
                weights_path=str(LORA_STORE_PATH / "user_001" / "acme_brand_v1.safetensors"),
                base_model="stabilityai/stable-diffusion-xl-base-1.0",
                lora_scale=0.85,
                training_images=24,
                created_at="2026-04-01T10:00:00Z",
                status="ready",
                version=1,
            ),
            LoRAModel(
                lora_id="lora_apex_fitness_v2",
                user_id="user_002",
                brand_name="Apex Fitness",
                trigger_word="apexfit",
                weights_path=str(LORA_STORE_PATH / "user_002" / "apex_fitness_v2.safetensors"),
                base_model="stabilityai/stable-diffusion-xl-base-1.0",
                lora_scale=0.75,
                training_images=40,
                created_at="2026-04-10T14:30:00Z",
                status="ready",
                version=2,
            ),
        ]
        for m in demos:
            self._store[m.lora_id] = m

    def get(self, lora_id: str) -> Optional[LoRAModel]:
        return self._store.get(lora_id)

    def get_by_user(self, user_id: str) -> list[LoRAModel]:
        return [m for m in self._store.values() if m.user_id == user_id]

    def register(self, model: LoRAModel):
        self._store[model.lora_id] = model

    def list_all(self) -> list[LoRAModel]:
        return list(self._store.values())


# ---------------------------------------------------------------------------
# Base Model Manager
# Loads the frozen SD model once and keeps it hot in memory
# ---------------------------------------------------------------------------

class BaseModelManager:
    """
    In production: loads StableDiffusionPipeline onto GPU.
    Simulated here to run without hardware.

    Key principle: the base model is loaded ONCE and shared across all requests.
    LoRA weights are injected and ejected per-request on top of this frozen base.
    This is what makes LoRA efficient — you're not reloading a 6GB model per user.
    """

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.model_id = model_id
        self._loaded = False
        self._active_lora: Optional[str] = None

    def load(self):
        """
        Production code:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
            ).to("cuda")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()
        """
        print(f"[BaseModelManager] Loading base model: {self.model_id}")
        time.sleep(0)  # simulated load
        self._loaded = True
        print("[BaseModelManager] Base model ready.")

    def inject_lora(self, weights_path: str, lora_scale: float, trigger_word: str):
        """
        Injects LoRA adapter weights on top of the frozen base model.

        Production code:
            self.pipe.load_lora_weights(weights_path)
            self.pipe.fuse_lora(lora_scale=lora_scale)

        Why fuse_lora instead of just load_lora_weights?
        fuse_lora merges the LoRA delta into the base weights mathematically.
        This is faster at inference time — no extra computation per forward pass.
        We unfuse after the request to restore the clean base model.
        """
        print(f"[BaseModelManager] Injecting LoRA: {weights_path} at scale {lora_scale}")
        self._active_lora = weights_path

    def eject_lora(self):
        """
        Removes LoRA weights and restores the frozen base model.

        Production code:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()

        This is CRITICAL for correctness. Without ejecting, the next user's
        request would run with the previous user's LoRA still active.
        """
        if self._active_lora:
            print(f"[BaseModelManager] Ejecting LoRA: {self._active_lora}")
            self._active_lora = None

    def run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> dict:
        """
        Runs the diffusion loop.

        Production code:
            generator = torch.Generator("cuda").manual_seed(seed)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return result.images[0]  # PIL Image
        """
        # Simulate latency (real SDXL on A100: ~4–8s for 30 steps)
        simulated_latency_ms = steps * 150 + 500
        return {
            "simulated": True,
            "latency_ms": simulated_latency_ms,
            "seed": seed,
        }


# ---------------------------------------------------------------------------
# LoRA Pipeline
# Orchestrates: inject → infer → eject → store → return
# ---------------------------------------------------------------------------

class LoRAPipeline:

    def __init__(self):
        self.registry = LoRARegistry()
        self.base_model = BaseModelManager()
        self.base_model.load()

    def generate(self, req: InferenceRequest) -> InferenceResult:
        request_id = f"req_{uuid.uuid4().hex[:10]}"
        seed = req.seed or int(time.time() * 1000) % (2**32)
        start_ms = int(time.time() * 1000)

        lora_model: Optional[LoRAModel] = None
        trigger_injected = False
        lora_scale_used = 0.0
        final_prompt = req.prompt

        # --- Step 1: Resolve LoRA model (if requested) ---
        if req.lora_id:
            lora_model = self.registry.get(req.lora_id)

            if not lora_model:
                return InferenceResult(
                    request_id=request_id,
                    user_id=req.user_id,
                    lora_id=req.lora_id,
                    trigger_word_injected=False,
                    lora_scale_used=0.0,
                    asset_url="",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    latency_ms=0,
                    seed_used=seed,
                    prompt_final=final_prompt,
                    status="failed",
                    error=f"LoRA model '{req.lora_id}' not found.",
                )

            if lora_model.status != "ready":
                return InferenceResult(
                    request_id=request_id,
                    user_id=req.user_id,
                    lora_id=req.lora_id,
                    trigger_word_injected=False,
                    lora_scale_used=0.0,
                    asset_url="",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    latency_ms=0,
                    seed_used=seed,
                    prompt_final=final_prompt,
                    status="failed",
                    error=f"LoRA model '{req.lora_id}' is not ready (status: {lora_model.status}).",
                )

            # --- Step 2: Inject trigger word into prompt ---
            # The trigger word is what activates the LoRA's learned features.
            # It must appear in the prompt or the LoRA has no effect.
            if lora_model.trigger_word not in final_prompt:
                final_prompt = f"{lora_model.trigger_word}, {final_prompt}"
                trigger_injected = True

            lora_scale_used = lora_model.lora_scale

            # --- Step 3: Inject LoRA weights into base model ---
            self.base_model.inject_lora(
                weights_path=lora_model.weights_path,
                lora_scale=lora_scale_used,
                trigger_word=lora_model.trigger_word,
            )

        # --- Step 4: Run diffusion inference ---
        try:
            raw = self.base_model.run_inference(
                prompt=final_prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=seed,
            )
        finally:
            # --- Step 5: ALWAYS eject LoRA — even on exception ---
            # This guard ensures the base model is never left in a dirty state.
            if lora_model:
                self.base_model.eject_lora()

        # --- Step 6: Build CDN URL and store asset ---
        asset_filename = f"{request_id}.png"
        lora_segment = lora_model.lora_id if lora_model else "base"
        asset_url = f"{CDN_BASE}/{req.user_id}/{lora_segment}/{asset_filename}"

        latency_ms = int(time.time() * 1000) - start_ms + raw.get("latency_ms", 0)

        return InferenceResult(
            request_id=request_id,
            user_id=req.user_id,
            lora_id=req.lora_id,
            trigger_word_injected=trigger_injected,
            lora_scale_used=lora_scale_used,
            asset_url=asset_url,
            created_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
            seed_used=seed,
            prompt_final=final_prompt,
            status="success",
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_pipeline: Optional[LoRAPipeline] = None


def get_pipeline() -> LoRAPipeline:
    """Singleton — base model loads once, stays in memory."""
    global _pipeline
    if _pipeline is None:
        _pipeline = LoRAPipeline()
    return _pipeline


def generate_personalised_image(payload: dict) -> dict:
    """
    Called by app.py.

    Expected keys:
        user_id              : str  (required)
        prompt               : str  (required)
        lora_id              : str  (optional — omit for base model generation)
        negative_prompt      : str  (optional)
        width / height       : int  (optional, default 1024)
        num_inference_steps  : int  (optional, default 30)
        guidance_scale       : float (optional, default 7.5)
        seed                 : int  (optional — set for reproducibility)
        style_preset         : str  (optional)
    """
    if not payload.get("user_id"):
        return {"error": "user_id is required", "status": "failed"}
    if not payload.get("prompt"):
        return {"error": "prompt is required", "status": "failed"}

    req = InferenceRequest(
        user_id=payload["user_id"],
        prompt=payload["prompt"],
        lora_id=payload.get("lora_id"),
        negative_prompt=payload.get("negative_prompt", "blurry, low quality, distorted, watermark"),
        width=payload.get("width", 1024),
        height=payload.get("height", 1024),
        num_inference_steps=payload.get("num_inference_steps", 30),
        guidance_scale=payload.get("guidance_scale", 7.5),
        seed=payload.get("seed"),
        style_preset=payload.get("style_preset", "photorealistic"),
    )

    pipeline = get_pipeline()
    result = pipeline.generate(req)
    return asdict(result)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = [
        {
            "label": "With LoRA (brand-consistent face/style)",
            "payload": {
                "user_id": "user_001",
                "prompt": "professional product shot of moisturiser on marble surface, studio lighting",
                "lora_id": "lora_acme_brand_v1",
                "seed": 42,
            },
        },
        {
            "label": "Base model only (no LoRA)",
            "payload": {
                "user_id": "user_003",
                "prompt": "hero banner for a fitness app, bold colours, dynamic athlete",
                "seed": 99,
            },
        },
        {
            "label": "Invalid LoRA ID",
            "payload": {
                "user_id": "user_001",
                "prompt": "product shot",
                "lora_id": "lora_does_not_exist",
            },
        },
    ]

    for s in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {s['label']}")
        print("="*60)
        result = generate_personalised_image(s["payload"])
        print(json.dumps(result, indent=2))