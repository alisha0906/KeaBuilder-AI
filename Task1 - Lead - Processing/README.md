# KeaBuilder AI Assignment

## Problem Statement

Design and implement an AI-powered lead processing system for KeaBuilder, a SaaS platform focused on funnels, lead capture, and automation.

The goal is to improve how incoming leads from forms are handled using AI by:

- classifying leads into HOT / WARM / COLD
- generating personalized human-like responses
- detecting hidden buying intent
- handling incomplete or unclear submissions
- recommending best follow-up strategies
- improving lead conversion efficiency

This assignment demonstrates practical AI engineering, prompt design, and system thinking.

---

# My Approach

Instead of building a basic lead classification system using only budget and timeline, I designed an:

# AI-Powered Sales Decision Engine

This system uses both:

## Explicit Signals

- budget
- project timeline
- decision-maker role
- company size
- form completeness
- requirement clarity

## Hidden Signals

- pain score
- frustration score
- urgency from natural language
- buying intent detection
- operational pain points
- competitor comparison signals

This makes the system much more practical for real SaaS sales teams.

---

# Unique Features Added

## 1. Hidden Buying Intent Detection

AI identifies urgency and seriousness even when users do not explicitly mention it.

Example:

“Manual inventory tracking is causing delays”

This indicates strong operational pain and may upgrade a lead from Warm to Hot.

---

## 2. Pain Score

Measures how severe the business problem is.

Higher pain = higher buying probability.

---

## 3. Frustration Score

Detects emotional urgency using phrases like:

- urgent
- struggling
- delays
- losing customers
- manual work is slowing us down

This improves lead prioritization.

---

## 4. Confidence Score

AI estimates confidence in its classification.

Low confidence triggers clarification questions instead of incorrect assumptions.

---

## 5. Recommended Sales Strategy

Instead of only classifying leads, AI also suggests:

- immediate demo booking
- send case study
- nurture via email
- senior consultant follow-up

This makes the system actionable for sales teams.

---

## 6. Recommended Communication Channel

AI decides the best response channel:

- Phone Call
- Email
- WhatsApp
- Follow-up sequence

This improves conversion efficiency.

---

# System Architecture

```text
Form Submission
        ↓
Data Validation
        ↓
Prompt 1 → Lead Classification Engine
        ↓
Lead Score + Pain Score + Frustration Score
        ↓
HOT / WARM / COLD Classification
        ↓
Prompt 2 → Response Generation Engine
        ↓
Personalized Human-like Response
        ↓
Recommended Strategy + Channel
        ↓
CRM Update + Sales Team Notification
```

# Tech Stack
### Backend
- Python
- FastAPI

### LLM Integration
- GROQ API
- Llama 3 Model

### Prompt Engineering
- JSON strured outputs
- Prompt-base classification
- Prompt-based response generation

### Environment Management
- python-dotenv

### Lead Classification Logic
# HOT Lead
- urgent requirement
- strong business pain
- high buying intent
- decision maker involved
- ready to purchase soon
- Example : “Need ERP within 2 weeks for factory operations”

# WARM Lead
- interested but still evaluating
- moderate urgency
- medium buying intent
- Example : “Exploring CRM options this quarter”

# COLD Lead
- unclear requirement
- low urgency
- only researching
- Example : “Just checking pricing for future use”

## Prompt Files
### classification_prompt.txt
Used for:
- lead classification
- score calculation
- pain/frustration detection

### response_prompt.txt

Used for:

- personalized responses
- urgency-aware messaging
- trust-building communication
When users submit vague forms like:

---

## Handling Incomplete Inputs

For vague inputs like:

“Need software for business”

AI detects missing details like:

- budget
- timeline
- use case
- company details

Instead of assuming, it asks smart follow-up questions.

---
## Run Project

### Install

```bash
pip install -r code/requirements.txt
```

### Add Groq API Key
Create .env
GROQ_API_KEY=your_actual_api_key_here

### Run FastAPI Server
```bash
cd code
uvicorn app:app --reload
```
### Test API

Open: http://127.0.0.1:8000/docs

Use:

POST /process-lead

This will generate AI-powered lead classification + response.
