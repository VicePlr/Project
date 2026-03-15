import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.face_recognizer import FaceRecognitionService, UserFaceDatabase
from classes.recognizer import OCRRecognizerBase


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GUEST_ID_PREFIX = "guest_"
GUEST_ID_WIDTH = 6


def resolve_path(*relative_parts):
    docker_path = os.path.join("/app", *relative_parts)
    local_path = os.path.join(PROJECT_ROOT, *relative_parts)
    return docker_path if os.path.exists(docker_path) else local_path


def build_face_services(users_csv_path: str):
    face_detector_model_path = resolve_path("models", "face", "det_10g.onnx")
    if not os.path.exists(face_detector_model_path):
        raise FileNotFoundError(
            f"Required SCRFD face detector model not found: {face_detector_model_path}"
        )
    face_recognizer_model_path = resolve_path(
        "models", "face", "face_recognition_sface_2021dec.onnx"
    )
    face_service = FaceRecognitionService(
        face_detector_model_path, face_recognizer_model_path
    )
    user_database = UserFaceDatabase(users_csv_path, face_service)
    return face_service, user_database


def safe_user_id(user_id: str) -> str:
    safe = "".join(char for char in user_id if char.isalnum() or char in ("-", "_"))
    return safe or "user"


def next_face_filename(user_faces_dir: str, source_path: str) -> str:
    extension = os.path.splitext(source_path)[1].lower() or ".jpg"
    max_index = 0
    if os.path.isdir(user_faces_dir):
        for entry in os.listdir(user_faces_dir):
            stem, current_extension = os.path.splitext(entry)
            if current_extension.lower() != extension:
                continue
            if stem.isdigit():
                max_index = max(max_index, int(stem))
    return f"{max_index + 1}{extension}"


def next_guest_user_id(users_csv: str) -> str:
    next_index = 1
    if os.path.exists(users_csv):
        with open(users_csv, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                current_user_id = row.get("user_id", "").strip().lower()
                if not current_user_id.startswith(GUEST_ID_PREFIX):
                    continue
                suffix = current_user_id[len(GUEST_ID_PREFIX) :]
                if suffix.isdigit():
                    next_index = max(next_index, int(suffix) + 1)
    return f"{GUEST_ID_PREFIX}{next_index:0{GUEST_ID_WIDTH}d}"


def enroll_user(
    user_id: str,
    plate: str,
    face_source_path: str,
    *,
    users_csv: str = None,
    faces_dir: str = None,
    is_video: bool = False,
    guest_mode: bool = False,
):
    users_csv = users_csv or resolve_path("data", "users", "users.csv")
    faces_dir = faces_dir or resolve_path("data", "users", "faces")

    if not os.path.exists(face_source_path):
        raise FileNotFoundError(f"Face source not found: {face_source_path}")

    resolved_user_id = user_id.strip() if user_id else ""
    if guest_mode or not resolved_user_id:
        resolved_user_id = next_guest_user_id(users_csv)

    normalized_plate = OCRRecognizerBase.normalize_vietnamese_plate(plate)
    _, user_database = build_face_services(users_csv)

    if not user_database.source_has_detectable_face(face_source_path):
        raise ValueError("No detectable face found in the provided face source.")

    user_faces_dir = os.path.join(faces_dir, safe_user_id(resolved_user_id))
    destination_filename = next_face_filename(user_faces_dir, face_source_path)
    destination_path = os.path.join(user_faces_dir, destination_filename)
    UserFaceDatabase.copy_face_source(face_source_path, destination_path)

    relative_face_dir = os.path.relpath(user_faces_dir, os.path.dirname(users_csv))
    user_database.upsert_user(
        resolved_user_id,
        normalized_plate,
        relative_face_dir,
    )

    return {
        "user_id": resolved_user_id,
        "registered_plate": normalized_plate,
        "face_dir": relative_face_dir.replace("\\", "/"),
        "face_source_path": os.path.relpath(
            destination_path, os.path.dirname(users_csv)
        ).replace("\\", "/"),
        "face_source_type": "video" if is_video else "image",
        "guest_mode": resolved_user_id.startswith(GUEST_ID_PREFIX),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enroll or update a user with user_id, a face image or video, and registered plate."
    )
    parser.add_argument("--user-id", help="Unique user identifier.")
    parser.add_argument("--plate", required=True, help="Registered license plate.")
    face_source_group = parser.add_mutually_exclusive_group(required=True)
    face_source_group.add_argument(
        "--face-image", help="Path to an enrolled face image."
    )
    face_source_group.add_argument(
        "--face-video", help="Path to an enrolled face video."
    )
    parser.add_argument(
        "--users-csv",
        default=resolve_path("data", "users", "users.csv"),
        help="CSV database path.",
    )
    parser.add_argument(
        "--faces-dir",
        default=resolve_path("data", "users", "faces"),
        help="Directory to store enrolled face images.",
    )
    parser.add_argument(
        "--guest",
        action="store_true",
        help="Auto-generate a guest user ID such as guest_000001.",
    )
    args = parser.parse_args()

    if not args.guest and not (args.user_id or "").strip():
        parser.error("--user-id is required unless --guest is used.")

    enrollment_result = enroll_user(
        args.user_id or "",
        args.plate,
        args.face_video or args.face_image,
        users_csv=args.users_csv,
        faces_dir=args.faces_dir,
        is_video=bool(args.face_video),
        guest_mode=args.guest,
    )
    print(json.dumps(enrollment_result, indent=2))


if __name__ == "__main__":
    main()
