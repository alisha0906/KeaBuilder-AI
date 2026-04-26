# Task 3 — Personalised AI Image Generation with LoRA
**KeaBuilder AI Assignment — Dream Reflection Media**

---

## What This Solves

KeaBuilder users need images that are consistent — same face, same brand palette,
same visual style — across every funnel, landing page, and campaign they build.
Standard image generation (base Stable Diffusion) can't guarantee this. Every
generation is independent; there's no "memory" of your brand.

**LoRA (Low-Rank Adaptation)** solves this by fine-tuning a small set of adapter
weights on top of a frozen base model using the user's own reference images.
The result: a personalised model that knows your brand's face, colours, and style —
and applies them consistently to any prompt.

---

## How LoRA Works (the concept)

A full diffusion model has billions of parameters. Retraining all of them per user
is prohibitively expensive. LoRA instead trains two small matrices (A and B) for
each layer, whose product approximates the weight delta:

```
ΔW = A × B     where rank(A×B) << rank(ΔW_full)
```

These matrices are tiny (~50–200 MB vs. 6 GB for the full model) and train fast
(~15–40 minutes on an A100 GPU vs. days for full fine-tuning). At inference time,
the adapter is injected mathematically on top of the frozen base — no reloading,
no copying, no separate model per user.

---

## User Journey Inside KeaBuilder

```
1. ONBOARDING
   User goes to "Brand Studio" in KeaBuilder
   Uploads 10–50 reference images (product photos, brand faces, style shots)
   Names their brand and sets a trigger word (e.g. "acmebrand")
         ↓
2. TRAINING (async, background GPU job)
   KeaBuilder validates images → queues training job
   Job runs DreamBooth LoRA training (~15–40 min)
   User gets email: "Your brand model is ready"
         ↓
3. GENERATION
   User types prompt in builder: "product shot on marble, warm lighting"
   Selects their brand from "My Brand Models" dropdown
   Hits Generate
         ↓
4. OUTPUT
   System injects trigger word → runs SDXL + LoRA → returns CDN URL
   Image appears in Media Library, tagged with brand + seed
   User can regenerate with same seed, adjust scale, or A/B test
```

---

## Architecture

```
Builder UI (Brand Studio + Image Generator)
            ↓
    POST /api/v1/images/generate
            ↓
       API Gateway (app.py)
            ↓
    Input validation
            ↓
    LoRA Registry lookup
    (user_id + lora_id → LoRAModel record)
            ↓
    Trigger word injection into prompt
            ↓
    BaseModelManager
    ├── inject_lora(weights_path, scale)   ← ~50ms
    ├── run_inference(prompt, ...)         ← ~4–8s on A100
    └── eject_lora()                      ← always runs, even on failure
            ↓
    CDN upload + Asset stored in Media Library
            ↓
    Response: asset_url, seed, lora_scale_used, trigger_word_injected
```

---

## LoRA Injection & Ejection — The Critical Design

The base model loads **once** and stays in GPU memory. LoRA weights are
**injected per-request** and **ejected immediately after**.

```python
# Inject
self.pipe.load_lora_weights(weights_path)
self.pipe.fuse_lora(lora_scale=lora_scale)

# Run inference
result = self.pipe(prompt=..., ...)

# Eject — ALWAYS, even on exception
self.pipe.unfuse_lora()
self.pipe.unload_lora_weights()
```

`fuse_lora()` mathematically merges the delta into the base weights for
the duration of the request — this is faster than applying it per-forward-pass.
`unfuse_lora()` reverses it exactly. Without the eject step, the next user's
request inherits the previous user's brand style. This is enforced with a
`try/finally` block so it cannot be skipped.

---

## Trigger Word Mechanics

Every LoRA model is trained with an instance prompt like:
```
"a photo of acmebrand"
```

The word `acmebrand` becomes the activation keyword — the LoRA's learned
features only "fire" when this word appears in the prompt.

If the user's prompt doesn't include their trigger word, the system injects it
automatically at the front:

```
User prompt:   "product shot on marble, warm lighting"
Final prompt:  "acmebrand, product shot on marble, warm lighting"
```

The response field `trigger_word_injected: true` signals this happened, so the
UI can show a notice and the user understands why outputs look brand-consistent.

---

## Key Parameters

| Parameter | Typical value | What it controls |
|---|---|---|
| `lora_rank` | 16 | Expressiveness of the adapter. Higher = more expressive, larger file. |
| `lora_alpha` | 32 | Scaling factor (usually 2× rank). Controls training stability. |
| `lora_scale` | 0.75–0.90 | Inference-time strength. 1.0 = full brand takeover; 0.5 = subtle influence. |
| `training_steps` | 1000–2000 | More steps = better quality up to a point; too many = overfitting. |
| `guidance_scale` | 7.5 | How strictly the model follows the prompt. |
| `seed` | any int | Fixed seed = reproducible output. Let user save seeds they like. |

---

## Training Pipeline

```
User uploads images
        ↓
validate_images()
  - Min 10, max 50 images
  - Supported formats only
  - In production: resolution check, deduplication
        ↓
Submit to GPU job queue (Celery + RunPod / Modal)
        ↓
accelerate launch train_dreambooth_lora_sdxl.py
  --instance_data_dir {images}
  --instance_prompt "a photo of {trigger_word}"
  --output_dir {output}
  --rank 16
  --max_train_steps 1000
        ↓
Output: {lora_id}.safetensors (~100 MB)
        ↓
Upload to S3 / permanent storage
        ↓
LoRARegistry.status → "ready"
        ↓
Notify user (email + in-app)
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/images/generate` | Generate image (with or without LoRA) |
| POST | `/api/v1/images/lora/train` | Submit LoRA training job |
| GET | `/api/v1/images/lora/<lora_id>` | Get LoRA model metadata |
| GET | `/api/v1/images/lora/user/<user_id>` | List all LoRAs for a user |

---

## Sample Input → Output

### Generate with LoRA
**Request:**
```json
{
  "user_id": "user_001",
  "prompt": "professional product shot of moisturiser on marble surface, studio lighting",
  "lora_id": "lora_acme_brand_v1",
  "seed": 42
}
```

**Response:**
```json
{
  "request_id": "req_e3b170de31",
  "user_id": "user_001",
  "lora_id": "lora_acme_brand_v1",
  "trigger_word_injected": true,
  "lora_scale_used": 0.85,
  "asset_url": "https://cdn.keabuilder.com/generated/user_001/lora_acme_brand_v1/req_e3b170de31.png",
  "created_at": "2026-04-26T10:19:51Z",
  "latency_ms": 5000,
  "seed_used": 42,
  "prompt_final": "acmebrand, professional product shot of moisturiser on marble surface, studio lighting",
  "status": "success"
}
```

---

---

## Running the Code

```bash
cd task3/code
pip install flask

# Run inference demo (no GPU needed — simulated)
python lora_pipeline.py

# Run training demo
python lora_trainer.py

# Run API server
python app.py  # → http://localhost:5001
```

---

## Production Considerations

**Async training:** The `/lora/train` endpoint should return `202 Accepted` immediately
and notify the user via webhook when the job completes. Training a LoRA takes
15–40 minutes — a synchronous HTTP response is not viable.

**GPU scheduling:** Use a dedicated GPU worker pool (RunPod, Modal, or AWS `p3` instances)
for training jobs. Inference can share a smaller pool of always-on A10G/A100 instances.

**LoRA caching:** For high-traffic users, pre-fuse their LoRA into a cached model variant
so injection overhead drops from ~50ms to ~0ms.

**Storage:** `.safetensors` files live in S3. The LoRA Registry (PostgreSQL) stores
metadata and the S3 path. Never store weights in the application database.

**Multi-LoRA:** Diffusers supports composing multiple LoRA adapters at different scales.
KeaBuilder could expose this as "Brand Style + Face LoRA" combinations for advanced users.

---

*Task 3 complete. See also: Task 1 (Lead Processing) and Task 2 (Multi-Provider Content Generation).*
