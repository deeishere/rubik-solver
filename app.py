"""Local web interface for the Rubik's cube scanner and solver."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import kociemba
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

from solvePython import COLOR_TO_FACE, FACES, class_to_face, cluster_axis

FACE_TO_COLOR = {face: color for color, face in COLOR_TO_FACE.items()}

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.onnx"
# The site lives in public/ (also what Cloudflare Pages serves).
app = Flask(__name__, static_folder="public", static_url_path="", template_folder="public")
model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else None


def detect_face(image_bytes: bytes) -> list[str]:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The camera image could not be read. Try capturing again.")
    result = model.predict(frame, conf=0.25, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            face = class_to_face(str(result.names[class_id]))
            if face is None:  # e.g. the "center" logo class
                continue
            color = FACE_TO_COLOR[face]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                ((x1 + x2) / 2, (y1 + y2) / 2, color, float(box.conf.item()))
            )
    if len(detections) != 9:
        raise ValueError(
            f"Detected {len(detections)} color stickers; expected 9. Keep one face visible, square to the camera, and try again."
        )
    cols = cluster_axis(np.array([d[0] for d in detections], dtype="float32"))
    rows = cluster_axis(np.array([d[1] for d in detections], dtype="float32"))
    grid = [[None] * 3 for _ in range(3)]
    for detection, row, col in zip(detections, rows, cols):
        old = grid[row][col]
        if old is None or detection[3] > old[1]:
            grid[row][col] = (detection[2], detection[3])
    if any(cell is None for row in grid for cell in row):
        raise ValueError(
            "The detections did not form a complete 3×3 grid. Recapture this face."
        )
    return [cell[0] for row in grid for cell in row]


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify({"model": model is not None})


@app.post("/api/detect")
def detect():
    if model is None:
        return jsonify(error=f"Model weights not found at {MODEL_PATH}"), 503
    if "image" not in request.files:
        return jsonify(error="No camera image was received."), 400
    try:
        return jsonify(face=detect_face(request.files["image"].read()))
    except ValueError as exc:
        return jsonify(error=str(exc)), 422
    except Exception as exc:
        app.logger.exception("Sticker detection failed")
        return jsonify(error=f"Detection failed: {exc}"), 500


@app.post("/api/solve")
def solve():
    payload = request.get_json(silent=True) or {}
    state = str(payload.get("state", "")).upper()
    if len(state) != 54 or any(char not in "URFDLB" for char in state):
        return jsonify(error="Enter exactly 54 face letters: U, R, F, D, L, B."), 400
    counts = Counter(state)
    if any(counts[face] != 9 for face in FACES):
        return jsonify(error="Each face color must appear exactly nine times."), 400
    for index, face in enumerate(FACES):
        if state[index * 9 + 4] != face:
            return jsonify(
                error=f"The center of face {face} must be {face}. Check its colors."
            ), 400
    try:
        solution = kociemba.solve(state)
        return jsonify(solution=solution, moves=solution.split())
    except Exception as exc:
        return jsonify(error=f"These stickers do not form a valid cube: {exc}"), 422


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
