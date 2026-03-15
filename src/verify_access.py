import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.access_verifier import AccessVerifier


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FACE_VIDEO_SAMPLE_FRAME_COUNT = 12
PLATE_VIDEO_SAMPLE_FRAME_COUNT = 12


def resolve_path(*relative_parts):
    docker_path = os.path.join("/app", *relative_parts)
    local_path = os.path.join(PROJECT_ROOT, *relative_parts)
    return docker_path if os.path.exists(docker_path) else local_path


def list_media_files(directory: str, allowed_extensions):
    if not os.path.isdir(directory):
        return []

    media_files = []
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            continue
        extension = os.path.splitext(entry)[1].lower()
        if extension in allowed_extensions:
            media_files.append(path)
    return media_files


def build_video_frame_label(checkpoint_dir: str, video_path: str, frame_index: int):
    return f"{relative_to_checkpoint(checkpoint_dir, video_path)}#frame={frame_index}"


def build_video_preview_name(video_path: str, frame_index: int):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"{stem}_frame_{frame_index:04d}.jpg"


def sample_video_frames(video_path: str, sample_count: int):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []

    frames = []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            selected_indices = np.linspace(
                0, max(frame_count - 1, 0), num=min(sample_count, frame_count), dtype=int
            ).tolist()
            selected_indices = sorted(set(int(index) for index in selected_indices))
            for frame_index in selected_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if success and frame is not None:
                    frames.append((frame_index, frame))
            return frames

        frame_index = 0
        while len(frames) < sample_count:
            success, frame = capture.read()
            if not success or frame is None:
                break
            frames.append((frame_index, frame))
            frame_index += 1
    finally:
        capture.release()

    return frames


def collect_checkpoint_inputs(checkpoint_dir: str, directory: str, sample_count: int):
    inputs = []
    for media_path in list_media_files(directory, IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
        extension = os.path.splitext(media_path)[1].lower()
        if extension in IMAGE_EXTENSIONS:
            inputs.append(
                {
                    "display_path": relative_to_checkpoint(checkpoint_dir, media_path),
                    "image_source": media_path,
                    "preprocessing_name": os.path.basename(media_path),
                }
            )
            continue

        for frame_index, frame in sample_video_frames(media_path, sample_count):
            inputs.append(
                {
                    "display_path": build_video_frame_label(
                        checkpoint_dir, media_path, frame_index
                    ),
                    "image_source": frame,
                    "preprocessing_name": build_video_preview_name(media_path, frame_index),
                }
            )
    return inputs


def build_verifier(users_csv_path: str):
    detector_model_path = resolve_path("models", "detector", "yolo_detector_model.pt")
    detector_config_path = resolve_path("models", "detector", "config.json")

    plate_ocr_model_path = resolve_path("models", "recognizer", "vietnamese_lp_ocr.pt")
    if not os.path.exists(plate_ocr_model_path):
        plate_ocr_model_path = resolve_path(
            "models", "recognizer", "license_plates_ocr_model.onnx"
        )

    plate_ocr_config_path = resolve_path(
        "models", "recognizer", "license_plates_ocr_config.yaml"
    )
    face_detector_model_path = resolve_path("models", "face", "det_10g.onnx")
    if not os.path.exists(face_detector_model_path):
        raise FileNotFoundError(
            f"Required SCRFD face detector model not found: {face_detector_model_path}"
        )
    face_recognizer_model_path = resolve_path(
        "models", "face", "face_recognition_sface_2021dec.onnx"
    )

    with open(detector_config_path, "r", encoding="utf-8") as config_file:
        detector_config = json.load(config_file)

    return AccessVerifier(
        plate_detector_model_path=detector_model_path,
        plate_detector_input_size=detector_config.get("input_size", 640),
        plate_ocr_model_path=plate_ocr_model_path,
        plate_ocr_config_path=plate_ocr_config_path,
        face_detector_model_path=face_detector_model_path,
        face_recognizer_model_path=face_recognizer_model_path,
        users_csv_path=users_csv_path,
    )


def write_result_json(checkpoint_id: str, result_payload: dict):
    checkpoint_dir = resolve_path("data", "incoming", checkpoint_id)
    output_path = os.path.join(checkpoint_dir, "result.json")
    with open(output_path, "w", encoding="utf-8") as result_file:
        json.dump(result_payload, result_file, indent=2)
    return output_path


def relative_to_checkpoint(checkpoint_dir: str, path: str) -> str:
    return os.path.relpath(path, checkpoint_dir).replace("\\", "/")


def build_failure_payload(checkpoint_id: str, reason: str):
    return {
        "checkpoint_id": checkpoint_id,
        "success": False,
        "reason": reason,
        "best_match": None,
    }


def ensure_preprocessing_dirs(checkpoint_dir: str):
    preprocessing_dir = os.path.join(checkpoint_dir, "preprocessing")
    if os.path.isdir(preprocessing_dir):
        shutil.rmtree(preprocessing_dir)
    face_bbox_dir = os.path.join(preprocessing_dir, "face_bbox")
    plate_dir = os.path.join(preprocessing_dir, "plate")
    os.makedirs(face_bbox_dir, exist_ok=True)
    os.makedirs(plate_dir, exist_ok=True)
    return preprocessing_dir, face_bbox_dir, plate_dir


def build_demo_best_match(best_match: dict):
    if best_match is None:
        return None

    demo_match = {
        "scanned_plate": best_match["scanned_plate"],
        "plate_score": best_match["scanned_plate_confidence"],
        "plate_matches": best_match["plate_matches"],
        "plate_image": best_match["plate_image"],
    }
    optional_fields = {
        "user_id": best_match.get("user_id"),
        "face_similarity": best_match.get("face_similarity"),
        "registered_plate": best_match.get("registered_plate"),
        "face_image": best_match.get("face_image"),
    }
    for key, value in optional_fields.items():
        if value is not None:
            demo_match[key] = value
    return demo_match


def verify_checkpoint(checkpoint_id: str, users_csv: str = None):
    users_csv = users_csv or resolve_path("data", "users", "users.csv")
    checkpoint_dir = resolve_path("data", "incoming", checkpoint_id)
    face_dir = os.path.join(checkpoint_dir, "face")
    plate_dir = os.path.join(checkpoint_dir, "plate")

    if not os.path.isdir(checkpoint_dir):
        result_payload = build_failure_payload(
            checkpoint_id,
            f"Checkpoint folder '{checkpoint_id}' does not exist.",
        )
        output_path = write_result_json(checkpoint_id, result_payload)
        return result_payload, output_path

    if not os.path.isdir(face_dir):
        result_payload = build_failure_payload(
            checkpoint_id,
            f"Missing face folder for checkpoint '{checkpoint_id}'.",
        )
        output_path = write_result_json(checkpoint_id, result_payload)
        return result_payload, output_path

    if not os.path.isdir(plate_dir):
        result_payload = build_failure_payload(
            checkpoint_id,
            f"Missing plate folder for checkpoint '{checkpoint_id}'.",
        )
        output_path = write_result_json(checkpoint_id, result_payload)
        return result_payload, output_path

    face_inputs = collect_checkpoint_inputs(
        checkpoint_dir, face_dir, FACE_VIDEO_SAMPLE_FRAME_COUNT
    )
    plate_inputs = collect_checkpoint_inputs(
        checkpoint_dir, plate_dir, PLATE_VIDEO_SAMPLE_FRAME_COUNT
    )

    if not face_inputs:
        result_payload = build_failure_payload(
            checkpoint_id,
            f"No usable face images or video frames found for checkpoint '{checkpoint_id}'.",
        )
        output_path = write_result_json(checkpoint_id, result_payload)
        return result_payload, output_path

    if not plate_inputs:
        result_payload = build_failure_payload(
            checkpoint_id,
            f"No usable plate images or video frames found for checkpoint '{checkpoint_id}'.",
        )
        output_path = write_result_json(checkpoint_id, result_payload)
        return result_payload, output_path

    verifier = build_verifier(users_csv)
    _, face_bbox_dir, plate_preprocessing_dir = ensure_preprocessing_dirs(checkpoint_dir)

    plate_results = []
    for plate_input in plate_inputs:
        detection_results = verifier.plate_detector.infer(plate_input["image_source"])
        verifier.plate_detector.process_results(
            detection_results,
            plate_input["preprocessing_name"],
            output_dir=plate_preprocessing_dir,
        )
        plate_scan = verifier.scan_plate(
            plate_input["image_source"], detection_results=detection_results
        )
        plate_results.append(
            {
                "plate_image": plate_input["display_path"],
                "scanned_plate": plate_scan.scanned_plate,
                "scanned_plate_confidence": plate_scan.scanned_plate_confidence,
                "readable": plate_scan.scanned_plate is not None,
                "reason": plate_scan.reason,
            }
        )

    readable_plates = [plate for plate in plate_results if plate["readable"]]
    if not readable_plates:
        result_payload = build_failure_payload(
            checkpoint_id,
            "No readable plates were found for this checkpoint.",
        )
        output_path = write_result_json(checkpoint_id, result_payload)
        return result_payload, output_path

    best_plate_result = max(
        readable_plates, key=lambda plate: plate["scanned_plate_confidence"] or -1.0
    )
    registered_user = verifier.lookup_registered_user_by_plate(best_plate_result["scanned_plate"])

    face_analyses = []
    for face_input in face_inputs:
        face_analysis = verifier.face_service.analyze_face(face_input["image_source"])
        face_analysis.source = face_input["display_path"]
        face_analyses.append(face_analysis)
        verifier.face_service.save_detection_preview(
            face_input["image_source"],
            os.path.join(face_bbox_dir, face_input["preprocessing_name"]),
            analysis=face_analysis,
        )

    if registered_user is None:
        success = False
        reason = "Scanned plate is not registered in the user database."
        best_match = {
            "user_id": None,
            "face_similarity": None,
            "registered_plate": None,
            "scanned_plate": best_plate_result["scanned_plate"],
            "scanned_plate_confidence": best_plate_result["scanned_plate_confidence"],
            "plate_matches": False,
            "face_image": None,
            "plate_image": best_plate_result["plate_image"],
        }
    else:
        face_match = verifier.match_face_analyses_for_registered_user(
            face_analyses, registered_user
        )
        representative_face_image = None
        if face_match.face_images_used:
            representative_face_image = face_match.face_images_used[0]

        best_match = {
            "user_id": registered_user["user_id"],
            "face_similarity": face_match.face_similarity,
            "registered_plate": registered_user["registered_plate"],
            "scanned_plate": best_plate_result["scanned_plate"],
            "scanned_plate_confidence": best_plate_result["scanned_plate_confidence"],
            "plate_matches": True,
            "face_image": representative_face_image,
            "plate_image": best_plate_result["plate_image"],
        }
        success = face_match.matched
        reason = (
            "Scanned plate matched a registered user and the face matched that user's face folder."
            if success
            else face_match.reason
        )

    result_payload = {
        "checkpoint_id": checkpoint_id,
        "success": success,
        "reason": reason,
        "best_match": build_demo_best_match(best_match),
    }
    output_path = write_result_json(checkpoint_id, result_payload)
    return result_payload, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Verify all face images and plate images for a checkpoint and write one result.json."
    )
    parser.add_argument(
        "--checkpoint-id",
        required=True,
        help="Checkpoint or lane identifier for this verification request.",
    )
    parser.add_argument(
        "--users-csv",
        default=resolve_path("data", "users", "users.csv"),
        help="CSV database with columns: user_id, registered_plate, face_dir",
    )
    args = parser.parse_args()

    _, output_path = verify_checkpoint(args.checkpoint_id, users_csv=args.users_csv)
    print(json.dumps({"result": "written", "result_file": output_path}, indent=2))


if __name__ == "__main__":
    main()
