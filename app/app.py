#!/usr/bin/env python3
"""Flask dashboard for the License Plate Recognition System.

Two views:
  * ``/``          — upload a photo, run detect+recognize, see the annotated
                     result and the predicted plate string(s) with confidence.
  * ``/dashboard`` — KPI / metrics presentation: headline numbers, learning
                     curve, per-length accuracy, confusion pairs, calibration
                     and a gallery of example predictions.

The heavy pipeline is loaded lazily on first request so the server starts fast
even before a model is trained (the upload page then shows a helpful message).
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import cv2
import numpy as np
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.config import load_config
from lpr.inference.pipeline import LPRPipeline

CFG = load_config()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = CFG.app.max_upload_mb * 1024 * 1024

UPLOAD_DIR = CFG.abspath(CFG.app.upload_dir)
RESULT_DIR = os.path.join(UPLOAD_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Lazily-initialised singleton pipeline (detector always available; recognizer
# loads on first successful use).
_pipeline: LPRPipeline | None = None


def get_pipeline() -> LPRPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LPRPipeline()
    return _pipeline


def model_ready() -> bool:
    return os.path.exists(CFG.abspath(CFG.recognizer.weights_path))


def load_metrics() -> dict | None:
    path = CFG.abspath(CFG.recognizer.metrics_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_history() -> dict | None:
    path = os.path.join(CFG.abspath(CFG.training.reports_dir), "training_history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@app.route("/")
def index():
    return render_template("index.html", model_ready=model_ready())


@app.route("/recognize", methods=["POST"])
def recognize():
    """Handle an uploaded image: run the pipeline, return JSON + result image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported type {ext}"}), 400
    if not model_ready():
        return jsonify({"error": "No trained recognizer yet. Run "
                                 "scripts/train_recognizer.py first."}), 503

    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    stamp = f"{int(time.time()*1000)}_{secure_filename(file.filename)}"
    cv2.imwrite(os.path.join(UPLOAD_DIR, stamp), image)

    pipeline = get_pipeline()
    output = pipeline.run(image)
    annotated = pipeline.annotate(image, output)
    result_name = f"result_{stamp}.jpg"
    cv2.imwrite(os.path.join(RESULT_DIR, result_name), annotated)

    payload = output.to_dict()
    payload["original_url"] = url_for("uploaded_file", filename=stamp)
    payload["result_url"] = url_for("result_file", filename=result_name)
    return jsonify(payload)


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        metrics=load_metrics(),
        history=load_history(),
        model_ready=model_ready(),
    )


@app.route("/api/metrics")
def api_metrics():
    """JSON feed powering the interactive dashboard charts."""
    return jsonify({
        "metrics": load_metrics(),
        "history": load_history(),
        "model_ready": model_ready(),
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/uploads/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.route("/data/<path:filename>")
def data_file(filename):
    """Serve dataset images referenced by the example gallery."""
    return send_from_directory(PROJECT_ROOT, filename)


if __name__ == "__main__":
    print(f"[app] http://{CFG.app.host}:{CFG.app.port}  "
          f"(model_ready={model_ready()})")
    app.run(host=CFG.app.host, port=CFG.app.port, debug=False)
