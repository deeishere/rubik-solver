"""Scan a 3x3 Rubik's Cube with a YOLO sticker-color model.

Hold one face nearly square to the camera. Capture faces in Kociemba order:
U, R, F, D, L, B. The program prints the resulting 54-character state.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FACES = ("U", "R", "F", "D", "L", "B")
COLOR_TO_FACE = {
    "white": "U",
    "yellow": "D",
    "green": "F",
    "blue": "B",
    "orange": "L",
    "red": "R",
}


def class_to_face(name: str) -> str | None:
    """Map a YOLO color class to the standard Kociemba face letter."""
    key = name.lower().strip().replace("_", " ").replace("-", " ")
    return COLOR_TO_FACE.get(key)


def cluster_axis(values, count=3):
    """Split sticker centers into three ordered bands (left-to-right/top-to-bottom)."""
    samples = values.astype("float32").reshape(-1, 1)
    if len(samples) < count:
        raise ValueError("Fewer than three sticker centers along an axis.")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _compactness, labels, centers = cv2.kmeans(
        samples, count, None, criteria, 8, cv2.KMEANS_PP_CENTERS
    )
    order = centers[:, 0].argsort()
    rank = {int(cluster): position for position, cluster in enumerate(order)}
    return [rank[int(label)] for label in labels.reshape(-1)]


def read_face(model: YOLO, frame, face_name: str, confidence: float):
    result = model.predict(frame, conf=confidence, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError("No stickers detected. Move closer and improve lighting.")

    names = result.names
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(names[class_id])
        color = class_to_face(class_name)
        if color is None:  # Ignore classes such as "cube face".
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(((x1 + x2) / 2, (y1 + y2) / 2, color, float(box.conf.item())))

    if len(detections) != 9:
        raise ValueError(
            f"Found {len(detections)} color stickers; expected 9. "
            "Keep one face visible and unobstructed, then capture again."
        )

    xs = np.array([item[0] for item in detections], dtype="float32")
    ys = np.array([item[1] for item in detections], dtype="float32")
    cols = cluster_axis(xs)
    rows = cluster_axis(ys)

    grid = [[None for _ in range(3)] for _ in range(3)]
    for (x, y, color, score), row, col in zip(detections, rows, cols):
        cell = (row, col)
        existing = grid[row][col]
        if existing is None or score > existing[1]:
            grid[row][col] = (color, score)

    if any(cell is None for row in grid for cell in row):
        raise ValueError(
            "Sticker detections did not form a complete 3×3 grid; recapture."
        )

    letters = [[cell[0] for cell in row] for row in grid]
    print(f"\nDetected face {face_name} (rows as seen by camera):")
    for row in letters:
        print(" ".join(row))
    return "".join("".join(row) for row in letters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="/Users/doa/rubik solver /best.onnx",
        help="YOLO .pt weights",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Minimum YOLO confidence"
    )
    parser.add_argument(
        "--output", default="cube_state.txt", help="Output file for the 54-letter state"
    )
    parser.add_argument(
        "--no-solve", action="store_true", help="Only scan and save the state"
    )
    args = parser.parse_args()

    if not Path(args.weights).is_file():
        sys.exit(
            f"Weights not found: {args.weights}\nPass the path with --weights /path/to/best.pt"
        )

    model = YOLO(args.weights)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1.")

    print("Scan faces in order U, R, F, D, L, B.")
    print(
        "For each face, use this adjacent face as the camera-top edge: U→B, R→U, F→U, D→F, L→U, B→U."
    )
    print(
        "Keep the face square to camera and that reference edge at the top; SPACE captures, Q quits."
    )
    state = []
    try:
        for face_number, face in enumerate(FACES, start=1):
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame could not be read.")
                view = frame.copy()
                cv2.putText(
                    view,
                    f"Face {face}: SPACE capture | Q quit",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Rubik Cube Scanner", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Scan cancelled.")
                    return
                if key == 32:
                    try:
                        face_state = read_face(model, frame, face, args.confidence)
                    except ValueError as exc:
                        print(f"\n{exc}\nTry again, or press Q to quit.")
                        continue
                    state.extend(face_state)
                    print(f"Captured face {face_number} of 6 ({face}).")
                    break

        state_string = "".join(state)
        Path(args.output).write_text(state_string + "\n", encoding="utf-8")
        print(f"\nCube state: {state_string}")
        print(f"Saved to: {Path(args.output).resolve()}")
        counts = Counter(state_string)
        if any(counts[color] != 9 for color in "URFDLB"):
            print("Cannot solve: every face color must appear exactly 9 times.")
            print("Review the scan and recapture any face that was read incorrectly.")
            return
        for i, face in enumerate(FACES):
            if state_string[i * 9 + 4] != face:
                print(
                    f"Cannot solve: the center of scanned {face} was detected as "
                    f"{state_string[i * 9 + 4]}. This scanner expects the standard "
                    "color scheme (white U, red R, green F, yellow D, orange L, blue B)."
                )
                return
        if args.no_solve:
            print("Scan checks passed. Run without --no-solve to calculate moves.")
            return
        try:
            import kociemba
        except ImportError:
            print(
                "Solver library missing. Install it with: python3 -m pip install kociemba"
            )
            return

        try:
            solution = kociemba.solve(state_string)
        except Exception as exc:
            print(f"The scanned colors do not form a valid cube state: {exc}")
            print(
                "A sticker may have been misread, or the face orientations may not line up."
            )
            return

        moves = solution.split()
        print(f"\nSolution ({len(moves)} moves): {solution}")
        print("\nDo these moves in order. Look directly at the named face:")
        print("  U = top, R = right, F = front, D = bottom, L = left, B = back")
        print(
            "  A bare letter means clockwise; ' means counterclockwise; 2 means a half-turn."
        )
        for number, move in enumerate(moves, start=1):
            suffix = move[1:]
            direction = {"": "clockwise", "'": "counterclockwise", "2": "180 degrees"}[
                suffix
            ]
            print(f"{number}. Turn the {move[0]} face {direction}.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
