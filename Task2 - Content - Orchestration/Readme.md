# Task 2 — Multi-Provider AI Content Generation System

## Overview

This module designs how KeaBuilder can support AI-powered content generation for multiple media types inside the platform:

- Images
- Videos
- Voice

Each content type is handled by a different provider while keeping the frontend experience simple for users.

The goal is to build a scalable backend system that routes requests intelligently, stores outputs properly, and allows users to reuse generated assets inside funnels, landing pages, and campaigns.

---

## Problem Statement

KeaBuilder users should be able to click **Generate Content** inside the builder UI and create:

- marketing images
- product demo videos
- voiceovers for campaigns

using a single interface.

Behind the scenes:

- Images → handled by one provider
- Videos → handled by another provider
- Voice → handled by another provider

The system should decide where to send each request and manage all generated outputs inside the platform.

---

## My Approach

Instead of directly integrating provider APIs across the codebase, I designed an:

## AI Content Orchestration System

This includes:

- Provider Routing Layer
- AI Orchestrator
- Asset Management System
- Smart Provider Selection
- Fallback Handling

This keeps the platform scalable, cost-efficient, and production-ready.

---

## Routing Logic

### Provider Mapping

| Content Type | Primary Provider |
|---|---|
| Image | Stability AI |
| Video | Runway ML |
| Voice | ElevenLabs |

### Smart Provider Selection

Provider selection also depends on:

- user plan (Free / Premium)
- quality required
- generation speed
- provider cost
- provider availability
- fallback support

### Example

- Free user → lower-cost provider
- Premium user → higher-quality provider
- Provider failure → automatic fallback provider

This improves reliability and reduces failed generations.

---

## Frontend → Backend Flow

User action inside Builder UI:

1. Click **Generate Content**
2. Select content type (Image / Video / Voice)
3. Enter prompt
4. Submit request

Example prompt:

`Create a product demo video for a skincare brand`

Frontend sends request to backend API.

---

## Backend Processing Flow

```text
Validate Input
      ↓
Check User Plan / Credits
      ↓
Routing Engine
      ↓
Provider API Call
      ↓
Store Output
      ↓
Return Asset URL + Metadata
```

This keeps frontend simple and backend flexible.

---

## Output Management

Generated content should be stored inside:

## KeaBuilder Media Library

This includes:

- generated images
- generated videos
- generated voice files

Each asset stores:

- asset ID
- provider used
- asset URL
- created timestamp
- usage history
- generation metadata

This allows users to:

- reuse assets
- edit later
- track generated content
- manage campaigns efficiently

---

## Unique Feature Added

## Asset Versioning

If a user regenerates content, the system creates:

- Version 1
- Version 2
- Version 3

instead of overwriting the original file.

This improves usability and campaign management.

---

## High-Level Architecture

```text
Builder UI
    ↓
API Gateway
    ↓
AI Orchestrator
    ↓
Routing Engine
    ↓
Provider Layer
(Image / Video / Voice)
    ↓
Asset Storage + CDN
    ↓
Media Library + CRM Integration
```

---

This makes KeaBuilder more practical as a real AI-powered SaaS platform.

