"""
KeaBuilder — AI Content Generation API
Task 2: API Gateway Entry Point

Run:  python app.py
Test: curl -X POST http://localhost:5000/api/v1/generate \
          -H "Content-Type: application/json" \
          -d @sample_input.json
"""

from flask import Flask, request, jsonify
from routing_logic import generate_content, get_asset_versions

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "KeaBuilder AI Content API"})


# ---------------------------------------------------------------------------
# Generate Content
# POST /api/v1/generate
# ---------------------------------------------------------------------------

@app.route("/api/v1/generate", methods=["POST"])
def generate():
    """
    Accepts a generation request and routes it to the correct AI provider.

    Body (JSON):
        content_type  : "image" | "video" | "voice"   (required)
        prompt        : str                             (required)
        user_plan     : "free" | "premium"              (required)
        style         : str                             (optional)
        duration      : str                             (optional)
        quality       : str                             (optional)
        asset_id      : str                             (optional — for versioning)
    """
    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({"error": "Request body must be valid JSON", "status": "failed"}), 400

    required = ["content_type", "prompt", "user_plan"]
    missing = [f for f in required if not payload.get(f)]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing)}",
            "status": "failed"
        }), 400

    result = generate_content(payload)

    if result.get("status") == "failed":
        return jsonify(result), 422

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# Get Asset Versions
# GET /api/v1/assets/<asset_id>/versions
# ---------------------------------------------------------------------------

@app.route("/api/v1/assets/<asset_id>/versions", methods=["GET"])
def asset_versions(asset_id: str):
    """
    Returns all versions of a generated asset.
    Supports the asset versioning feature (Task 2 unique addition).
    """
    versions = get_asset_versions(asset_id)

    if not versions:
        return jsonify({"error": f"Asset '{asset_id}' not found", "status": "failed"}), 404

    return jsonify({
        "asset_id": asset_id,
        "total_versions": len(versions),
        "versions": versions,
    }), 200


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)