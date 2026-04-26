from fastapi import FastAPI
from lead_classifier import classify_lead_with_llm
from response_generator import generate_response_with_llm
import json
from pathlib import Path

app = FastAPI(title="KeaBuilder AI Lead Processing System")

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "sample_input.json"


def load_sample_input():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/")
def home():
    return {
        "message": "KeaBuilder AI Lead Processing System using Groq API is running"
    }


@app.post("/process-lead")
def process_lead():
    lead_data = load_sample_input()

    classification_result = classify_lead_with_llm(lead_data)
    response_result = generate_response_with_llm(
        lead_data,
        classification_result
    )

    final_output = {
        **classification_result,
        "response": response_result
    }

    return final_output