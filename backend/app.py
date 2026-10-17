import json
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

from engine import DiseaseModel, display_label  # noqa: F401  (DiseaseModel needed to unpickle)

BASE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE / "model" / "model.joblib"))
DEFAULT_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.80))
DEFAULT_MAX_QUESTIONS = int(os.environ.get("MAX_QUESTIONS", 7))
MIN_GAIN = 0.005  # stop asking when no question is informative

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.environ.get("CORS_ORIGINS", "*").split(",")}})

# ------------------------------------------------------------ load model
MODEL = None
MODEL_ERROR = ""
if MODEL_PATH.exists():
    MODEL = joblib.load(MODEL_PATH)
else:
    MODEL_ERROR = "Model not found. Run `python train_model.py` in the backend folder first."

INFO_PATH = BASE / "disease_info.json"
DISEASE_INFO = json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else {}
GENERIC_PRECAUTIONS = [
    "Talk to a doctor about these symptoms",
    "Do not start prescription medicine without medical advice",
    "Get urgent care if symptoms suddenly get worse",
]
URGENT_SYMPTOMS = {
    "sharp chest pain", "difficulty breathing", "vomiting blood", "hemoptysis",
    "fainting", "difficulty speaking", "rectal bleeding", "blood in stool",
}
DISCLAIMER = "This is not a medical diagnosis. Please consult a qualified doctor."


def title(name):
    return name[:1].upper() + name[1:]


def describe(name, probability):
    info = DISEASE_INFO.get(name, {})
    return {
        "disease": title(name),
        "probability": round(probability * 100, 1),
        "description": info.get("description", ""),
        "precautions": info.get("precautions") or GENERIC_PRECAUTIONS,
        "specialist": info.get("specialist") or "General physician",
    }


def clean_list(value):
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("Symptom lists must be arrays of strings.")
    return list(dict.fromkeys(value))


@app.get("/api/health")
def health():
    if MODEL is None:
        return jsonify(status="no_model", error=MODEL_ERROR), 503
    return jsonify(status="ok", diseases=len(MODEL.classes), symptoms=len(MODEL.symptoms))


@app.get("/api/symptoms")
def symptoms():
    if MODEL is None:
        return jsonify(error=MODEL_ERROR), 503
    popular = set(MODEL.prevalence.argsort()[::-1][:24].tolist())
    items = [{"id": s, "label": display_label(s), "popular": i in popular}
             for i, s in enumerate(MODEL.symptoms)]
    return jsonify(sorted(items, key=lambda x: x["label"]))


@app.post("/api/assess")
def assess():
    """Stateless adaptive assessment.

    Body: {confirmed: [...], denied: [...], skipped: [...], asked: int,
           threshold?: float, max_questions?: int}
    """
    if MODEL is None:
        return jsonify(error=MODEL_ERROR), 503
    body = request.get_json(silent=True) or {}
    try:
        confirmed = clean_list(body.get("confirmed"))
        denied = clean_list(body.get("denied"))
        skipped = clean_list(body.get("skipped"))
        asked = max(0, int(body.get("asked", 0)))
        threshold = min(0.99, max(0.5, float(body.get("threshold", DEFAULT_THRESHOLD))))
        max_q = min(15, max(0, int(body.get("max_questions", DEFAULT_MAX_QUESTIONS))))
    except (TypeError, ValueError) as e:
        return jsonify(error=str(e) or "Invalid request."), 400

    if not confirmed:
        return jsonify(error="Select at least one symptom you have."), 400
    unknown = [s for s in confirmed + denied + skipped if s not in MODEL.index]
    if unknown:
        return jsonify(error=f"Unknown symptoms: {', '.join(unknown[:5])}"), 400
    if set(confirmed) & set(denied):
        return jsonify(error="A symptom cannot be both confirmed and denied."), 400

    state = MODEL.state_from(confirmed, denied)
    probs = MODEL.predict_proba(state)[0]
    top = probs.argsort()[::-1][:3]
    predictions = [describe(MODEL.classes[i], float(probs[i])) for i in top]

    finished, reason, question = False, None, None
    if probs[top[0]] >= threshold:
        finished, reason = True, "confidence"
    elif asked >= max_q:
        finished, reason = True, "max_questions"
    else:
        answered = [MODEL.index[s] for s in confirmed + denied + skipped]
        q = MODEL.next_question(state, probs, answered)
        if q is None or q["info_gain"] < MIN_GAIN:
            finished, reason = True, "no_questions"
        else:
            question = {"symptom": q["symptom"], "label": display_label(q["symptom"]),
                        "info_gain": round(q["info_gain"], 4)}

    urgent = sorted(s for s in confirmed if s in URGENT_SYMPTOMS)
    return jsonify(
        predictions=predictions,
        next_question=question,
        finished=finished,
        reason=reason,
        threshold=threshold,
        urgent=({"symptoms": [display_label(s) for s in urgent],
                 "message": "Some of your symptoms can sometimes signal an emergency. If they are severe or sudden, get medical help right away."}
                if urgent else None),
        disclaimer=DISCLAIMER,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
