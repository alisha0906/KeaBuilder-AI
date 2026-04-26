# System Design — Task 2
## Multi-Provider AI Content Generation

This document covers the detailed technical design for Task 2.
For the executive summary and usage instructions, see `README.md`.

---

## Component Breakdown

### 1. API Gateway (`app.py`)

Responsibilities:
- Receives POST requests from the Builder UI
- Validates required fields (`content_type`, `prompt`, `user_plan`)
- Delegates to the Routing Engine
- Returns structured JSON responses

Endpoints:
- `POST /api/v1/generate` — trigger content generation
- `GET  /api/v1/assets/<id>/versions` — list asset versions
- `GET  /health` — liveness check

### 2. Routing Engine (`routing_logic.py` → `RoutingEngine`)

Responsibilities:
- Selects the correct provider chain from `ROUTING_TABLE`
- Runs `health_check()` on each provider before calling it
- Implements automatic fallback if primary is unavailable
- Delegates to the Asset Store after successful generation

Key data structures:
```python
ROUTING_TABLE = {
  ("image", "premium"):  ["stability_ai", "dalle"],
  ("image", "free"):     ["dalle", "stability_ai"],
  ("video", "premium"):  ["runway_ml", "pika_labs"],
  ("video", "free"):     ["pika_labs", "runway_ml"],
  ("voice", "premium"):  ["elevenlabs", "openai_tts"],
  ("voice", "free"):     ["openai_tts", "elevenlabs"],
}
```

### 3. Provider Layer

Each provider extends `ContentProvider` and implements:
- `generate(prompt, options) → dict` — calls the provider's API
- `health_check() → bool` — checks provider availability

In production, `generate()` wraps the real HTTP call to the provider.
The base class handles the response schema, so adding a new provider
requires only subclassing and implementing these two methods.

Current providers:

| Provider | Content | Plan tier | Cost/call |
|---|---|---|---|
| Stability AI | Image | Premium | $0.04 |
| DALL·E | Image | Free / Fallback | $0.02 |
| Runway ML | Video | Premium | $0.50 |
| Pika Labs | Video | Free / Fallback | $0.20 |
| ElevenLabs | Voice | Premium | $0.03 |
| OpenAI TTS | Voice | Free / Fallback | $0.01 |

### 4. Asset Store

In this implementation: in-memory dict (`_asset_store`).
In production: replace with PostgreSQL or DynamoDB table.

Schema per asset record:
```json
{
  "asset_id":    "vid_53cf37e7",
  "version":     1,
  "content_type": "video",
  "provider":    "runway_ml",
  "asset_url":   "https://cdn.keabuilder.com/assets/vid_53cf37e7/v1/...",
  "created_at":  "2026-04-26T08:48:34Z",
  "prompt":      "...",
  "options":     {},
  "cost_usd":    0.50,
  "used_in":     [],
  "generation_metadata": {}
}
```

Versioning: each `asset_id` maps to a list of version records.
Regenerating with the same `asset_id` appends `v2`, `v3`, etc.

---

## Request → Response Flow (detailed)

```
1. Builder UI sends POST /api/v1/generate
           ↓
2. app.py validates fields (returns 400 if missing)
           ↓
3. generate_content(payload) called
           ↓
4. RoutingEngine.route(content_type, user_plan, prompt, options)
           ↓
5. Look up provider_chain from ROUTING_TABLE
           ↓
6. For each provider in chain:
     a. provider.health_check()  → skip if False
     b. provider.generate(...)   → use result if success
     c. On exception → try next provider
           ↓
7. If all providers fail → raise RoutingError → return 422
           ↓
8. store_asset(asset_id, version, payload) → persists record
           ↓
9. Build response dict with asset_url, fallback_used, version
           ↓
10. Return 200 JSON to Builder UI
```

---

## Failure Modes and Mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Provider API down | `health_check()` returns False | Automatic fallback to next provider |
| Provider returns error | Exception in `generate()` | Caught, logged, fallback triggered |
| All providers unavailable | `RoutingError` raised | Returns `status: failed` with error message |
| Invalid content type | Validated in `route()` | Returns `status: failed`, no API call made |
| Missing prompt | Validated in `app.py` | Returns 400 before routing engine runs |

---

## What Would Change in Production

1. **Async calls** — `generate()` should be `async def` using `aiohttp` or `httpx`, especially for video which takes 30–60s
2. **Queue-based processing** — video generation should be handled via a job queue (Celery + Redis) with webhook callbacks rather than a synchronous HTTP response
3. **Real health checks** — each provider's status page or `/ping` endpoint polled every 60s and cached
4. **Database** — replace `_asset_store` dict with a proper ORM model
5. **Credit system** — deduct user credits before calling provider; refund on failure
6. **CDN upload** — after `generate()`, upload the file to S3/Cloudflare and return the CDN URL, not the provider's URL
7. **Observability** — log `provider_used`, `fallback_used`, `cost_usd`, `latency_ms` per request for cost dashboards and SLA monitoring