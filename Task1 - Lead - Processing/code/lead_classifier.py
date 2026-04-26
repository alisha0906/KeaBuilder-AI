import json
from pathlib import Path
from groq import Groq

client = Groq(
    api_key="YOUR_GROQ_API_KEY"
)

MODEL_NAME = "llama3-8b-8192"

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = BASE_DIR / "prompts" / "classification_prompt.txt"


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as file:
        return file.read()


def clean_json_response(raw_response: str):
    raw_response = raw_response.strip()

    if raw_response.startswith("```json"):
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()

    elif raw_response.startswith("```"):
        raw_response = raw_response.replace("```", "").strip()

    return raw_response


def classify_lead_with_llm(lead_data: dict):
    prompt_template = load_prompt()

    final_prompt = prompt_template.replace(
        "{{FORM_INPUT}}",
        json.dumps(lead_data, indent=2)
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0.2
    )

    raw_response = response.choices[0].message.content
    cleaned_response = clean_json_response(raw_response)

    try:
        return json.loads(cleaned_response)

    except json.JSONDecodeError:
        return {
            "lead_category": "WARM",
            "lead_score": 50,
            "pain_score": 0,
            "frustration_score": 0,
            "confidence_score": 50,
            "buying_intent_detected": False,
            "reasoning": "Fallback due to invalid LLM JSON output.",
            "missing_information": [],
            "recommended_strategy": "Manual review required",
            "recommended_channel": "Email"
        }