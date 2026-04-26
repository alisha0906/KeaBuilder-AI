"""
KeaBuilder — LoRA Training Pipeline
Task 3: How a user's uploaded reference images become a LoRA model

This module handles:
  - Validating uploaded reference images
  - Triggering a training job (async, not blocking the user)
  - Updating the LoRA registry when training completes

In production this runs as a background worker (Celery + GPU instance).
Here it is simulated synchronously for demo purposes.
"""

import uuid
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from lora_pipeline import LoRAModel, LoRARegistry, LORA_STORE_PATH

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMAGES = 10
MAX_IMAGES = 50
MIN_RESOLUTION = 512   # px per side


@dataclass
class TrainingRequest:
    user_id: str
    brand_name: str
    trigger_word: str
    image_paths: list[str]          # paths to uploaded reference images
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_rank: int = 16             # higher = more expressive but larger file
    lora_alpha: int = 32            # scaling factor; typically 2x rank
    training_steps: int = 1000      # 1000–2000 for faces/brand styles
    learning_rate: float = 1e-4
    lora_scale: float = 0.80        # default inference strength


@dataclass
class TrainingResult:
    lora_id: str
    user_id: str
    brand_name: str
    trigger_word: str
    weights_path: str
    status: str
    training_steps_completed: int
    training_time_seconds: float
    created_at: str
    error: Optional[str] = None


def validate_images(image_paths: list[str]) -> tuple[bool, str]:
    """
    Checks images before kicking off a training job.
    Fails fast to avoid wasting GPU time.

    Rules:
    - At least MIN_IMAGES, at most MAX_IMAGES
    - All supported formats
    - In production: also check resolution >= MIN_RESOLUTION x MIN_RESOLUTION
      and reject images that are too similar (deduplication)
    """
    if len(image_paths) < MIN_IMAGES:
        return False, f"Need at least {MIN_IMAGES} images, got {len(image_paths)}."
    if len(image_paths) > MAX_IMAGES:
        return False, f"Max {MAX_IMAGES} images allowed, got {len(image_paths)}."

    for path in image_paths:
        suffix = Path(path).suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {path}. Allowed: {SUPPORTED_FORMATS}"

    return True, "ok"


def trigger_training_job(req: TrainingRequest) -> TrainingResult:
    """
    In production:
        1. Upload images to S3 staging bucket
        2. Submit job to GPU worker queue (Celery / Modal / RunPod)
        3. Worker runs:
               accelerate launch train_dreambooth_lora_sdxl.py \
                   --model_name_or_path {req.base_model} \
                   --instance_data_dir {image_dir} \
                   --instance_prompt "a photo of {req.trigger_word}" \
                   --output_dir {output_dir} \
                   --rank {req.lora_rank} \
                   --max_train_steps {req.training_steps} \
                   --learning_rate {req.learning_rate}
        4. On completion: upload .safetensors to permanent storage
        5. Webhook / callback updates LoRARegistry status → "ready"
        6. Notify user via email / in-app notification

    Here: simulated synchronously.
    """

    valid, msg = validate_images(req.image_paths)
    if not valid:
        return TrainingResult(
            lora_id="",
            user_id=req.user_id,
            brand_name=req.brand_name,
            trigger_word=req.trigger_word,
            weights_path="",
            status="failed",
            training_steps_completed=0,
            training_time_seconds=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            error=msg,
        )

    lora_id = f"lora_{req.trigger_word.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    weights_path = str(
        LORA_STORE_PATH / req.user_id / f"{lora_id}.safetensors"
    )

    # Simulate training time (real: ~15–40 min on A100 for 1000 steps)
    simulated_seconds = req.training_steps * 0.001
    time.sleep(0)

    result = TrainingResult(
        lora_id=lora_id,
        user_id=req.user_id,
        brand_name=req.brand_name,
        trigger_word=req.trigger_word,
        weights_path=weights_path,
        status="ready",
        training_steps_completed=req.training_steps,
        training_time_seconds=simulated_seconds,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Register the trained model so it's immediately available for inference
    registry = LoRARegistry()
    registry.register(LoRAModel(
        lora_id=lora_id,
        user_id=req.user_id,
        brand_name=req.brand_name,
        trigger_word=req.trigger_word,
        weights_path=weights_path,
        base_model=req.base_model,
        lora_scale=req.lora_scale,
        training_images=len(req.image_paths),
        created_at=result.created_at,
        status="ready",
        version=1,
    ))

    return result


if __name__ == "__main__":
    # Demo: train a new LoRA for a fictional brand
    req = TrainingRequest(
        user_id="user_005",
        brand_name="NovaDerm Cosmetics",
        trigger_word="novaderm",
        image_paths=[f"uploads/user_005/ref_{i:02d}.jpg" for i in range(1, 21)],
        training_steps=1000,
    )

    print("Submitting training job...")
    result = trigger_training_job(req)
    print(json.dumps(asdict(result), indent=2))