"""
KeaBuilder — Personalised Image Generation API
Task 3: API gateway exposing LoRA inference and training endpoints

Endpoints:
  POST /api/v1/images/generate        — generate image (with or without LoRA)
  POST /api/v1/images/lora/train      — submit LoRA training job
  GET  /api/v1/images/lora/<lora_id>  — get LoRA model metadata
  GET  /api/v1/images/lora/user/<uid> — list all LoRAs for a user
  GET  /health
"""

from flask import Flask, request, jsonify
from lora_pipeline import generate_personalised_image, get_pipeline
from lora_trainer import TrainingRequest, trigger_training_job
from dataclasses import asdict

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "KeaBuilder Personalised Image API"})


@app.route("/api/v1/images/generate", methods=["POST"])
def generate():
    """
    Generate a personalised image.
    Pass lora_id to use a brand-specific LoRA; omit for base model.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON", "status": "failed"}), 400

    for field in ("user_id", "prompt"):
        if not payload.get(field):
            return jsonify({"error": f"'{field}' is required", "status": "failed"}), 400

    result = generate_personalised_image(payload)

    if result.get("status") == "failed":
        return jsonify(result), 422

    return jsonify(result), 200


@app.route("/api/v1/images/lora/train", methods=["POST"])
def train_lora():
    """
    Submit a LoRA training job.
    In production this queues an async GPU job and returns immediately.
    The user is notified via webhook/email when training completes.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    for field in ("user_id", "brand_name", "trigger_word", "image_paths"):
        if not payload.get(field):
            return jsonify({"error": f"'{field}' is required"}), 400

    req = TrainingRequest(
        user_id=payload["user_id"],
        brand_name=payload["brand_name"],
        trigger_word=payload["trigger_word"],
        image_paths=payload["image_paths"],
        training_steps=payload.get("training_steps", 1000),
        lora_rank=payload.get("lora_rank", 16),
        lora_scale=payload.get("lora_scale", 0.80),
    )

    result = trigger_training_job(req)

    if result.status == "failed":
        return jsonify(asdict(result)), 422

    return jsonify(asdict(result)), 200


@app.route("/api/v1/images/lora/<lora_id>", methods=["GET"])
def get_lora(lora_id: str):
    pipeline = get_pipeline()
    model = pipeline.registry.get(lora_id)
    if not model:
        return jsonify({"error": f"LoRA '{lora_id}' not found"}), 404
    return jsonify(asdict(model)), 200


@app.route("/api/v1/images/lora/user/<user_id>", methods=["GET"])
def list_user_loras(user_id: str):
    pipeline = get_pipeline()
    models = pipeline.registry.get_by_user(user_id)
    return jsonify({
        "user_id": user_id,
        "total": len(models),
        "lora_models": [asdict(m) for m in models],
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)