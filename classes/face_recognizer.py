import csv
import shutil
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from classes.face_detector import create_face_detector
from classes.recognizer import OCRRecognizerBase


@dataclass
class UserFaceRecord:
    user_id: str
    registered_plate: str
    face_image_path: str
    face_dir: str
    embeddings: List[np.ndarray]


@dataclass
class FaceAnalysisResult:
    source: Optional[str]
    image: np.ndarray
    face: Optional[np.ndarray]
    embeddings: List[np.ndarray]
    detection_confidence: Optional[float]
    face_area_ratio: Optional[float]
    sharpness: Optional[float]
    brightness: Optional[float]
    center_score: Optional[float]
    quality_score: Optional[float]


class FaceRecognitionService:
    def __init__(
        self,
        detector_model_path: str,
        recognizer_model_path: str,
        score_threshold: float = 0.363,
    ):
        self.score_threshold = score_threshold
        self.detector = create_face_detector(detector_model_path)
        self.recognizer = cv2.FaceRecognizerSF_create(recognizer_model_path, "")

    @staticmethod
    def _load_image(image_source):
        if isinstance(image_source, str):
            image = cv2.imread(image_source, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image '{image_source}'.")
            return image
        if isinstance(image_source, np.ndarray):
            return image_source.copy()
        raise TypeError("Face image source must be a file path or numpy array.")

    def _detect_largest_face(self, image: np.ndarray):
        original_height, original_width = image.shape[:2]

        # Try a few scaled views first so large frames still yield a usable face.
        for max_side in (640, 960, 1280, None):
            if max_side is None or max(original_width, original_height) <= max_side:
                resized = image
                scale = 1.0
            else:
                scale = max_side / float(max(original_width, original_height))
                resized = cv2.resize(
                    image,
                    (int(original_width * scale), int(original_height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            faces = self.detector.detect(resized)
            if faces is None or len(faces) == 0:
                continue

            largest_face = max(faces, key=lambda face: float(face[2] * face[3])).copy()
            if scale != 1.0:
                largest_face[:14] = largest_face[:14] / scale
            return largest_face

        return None

    @staticmethod
    def _generate_detection_variants(image: np.ndarray) -> List[np.ndarray]:
        variants = [image]

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants.append(
            cv2.cvtColor(
                cv2.merge((clahe.apply(l_channel), a_channel, b_channel)),
                cv2.COLOR_LAB2BGR,
            )
        )
        variants.append(cv2.convertScaleAbs(image, alpha=1.1, beta=10))

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        variants.append(cv2.filter2D(image, -1, sharpen_kernel))
        return variants

    @staticmethod
    def _face_priority(face: np.ndarray) -> tuple:
        score = float(face[14]) if len(face) > 14 else 0.0
        area = float(face[2] * face[3])
        return score, area

    @staticmethod
    def _enhance_face_crop(face_crop: np.ndarray) -> List[np.ndarray]:
        variants = [face_crop]
        height, width = face_crop.shape[:2]

        # Extra crops help when helmets add a lot of non-face pixels around the
        # cheeks, forehead, and ears. We keep the center face area and let the
        # matcher choose the strongest embedding.
        centered_crop = face_crop[
            int(height * 0.08) : int(height * 0.96),
            int(width * 0.10) : int(width * 0.90),
        ]
        if centered_crop.size > 0:
            variants.append(cv2.resize(centered_crop, (width, height)))

        lower_face_crop = face_crop[
            int(height * 0.16) : int(height * 1.00),
            int(width * 0.12) : int(width * 0.88),
        ]
        if lower_face_crop.size > 0:
            variants.append(cv2.resize(lower_face_crop, (width, height)))

        # Helmet-heavy captures often preserve the nose, mouth, and jawline
        # more reliably than the forehead and temple regions.
        lower_tight_crop = face_crop[
            int(height * 0.22) : int(height * 1.00),
            int(width * 0.18) : int(width * 0.82),
        ]
        if lower_tight_crop.size > 0:
            variants.append(cv2.resize(lower_tight_crop, (width, height)))

        # Helmet shots often leave only part of the face visible, so mild contrast
        # recovery and sharpening can produce a more stable embedding.
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        contrast_variant = cv2.cvtColor(
            cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
        variants.append(contrast_variant)

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened_variant = cv2.filter2D(contrast_variant, -1, sharpen_kernel)
        variants.append(sharpened_variant)

        unique_variants = []
        for variant in variants:
            if not any(np.array_equal(variant, existing) for existing in unique_variants):
                unique_variants.append(variant)
        return unique_variants

    @staticmethod
    def _center_score(image_shape: tuple, face: np.ndarray) -> float:
        image_height, image_width = image_shape[:2]
        x, y, width, height = map(float, face[:4])
        face_center_x = x + width / 2.0
        face_center_y = y + height / 2.0
        normalized_dx = abs(face_center_x - (image_width / 2.0)) / max(image_width / 2.0, 1.0)
        normalized_dy = abs(face_center_y - (image_height / 2.0)) / max(
            image_height / 2.0, 1.0
        )
        distance = np.sqrt(normalized_dx * normalized_dx + normalized_dy * normalized_dy)
        return float(max(0.0, 1.0 - min(distance, 1.0)))

    @staticmethod
    def _brightness_score(brightness: float) -> float:
        return float(max(0.0, 1.0 - min(abs(brightness - 140.0) / 140.0, 1.0)))

    def _calculate_face_quality(
        self, image: np.ndarray, face: np.ndarray, aligned_face: np.ndarray
    ) -> dict:
        image_height, image_width = image.shape[:2]
        face_width = max(float(face[2]), 1.0)
        face_height = max(float(face[3]), 1.0)
        face_area_ratio = float((face_width * face_height) / max(image_height * image_width, 1))
        detection_confidence = float(face[14]) if len(face) > 14 else 0.0

        aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(aligned_gray, cv2.CV_32F).var())
        brightness = float(aligned_gray.mean())
        center_score = self._center_score(image.shape, face)

        normalized_area = min(face_area_ratio / 0.08, 1.0)
        normalized_sharpness = min(sharpness / 140.0, 1.0)
        brightness_score = self._brightness_score(brightness)
        quality_score = (
            0.35 * detection_confidence
            + 0.20 * normalized_area
            + 0.25 * normalized_sharpness
            + 0.10 * brightness_score
            + 0.10 * center_score
        )

        return {
            "detection_confidence": detection_confidence,
            "face_area_ratio": face_area_ratio,
            "sharpness": sharpness,
            "brightness": brightness,
            "center_score": center_score,
            "quality_score": float(quality_score),
        }

    @staticmethod
    def _analysis_priority(metrics: dict, face: np.ndarray) -> tuple:
        return (
            metrics["quality_score"],
            metrics["detection_confidence"],
            metrics["face_area_ratio"],
            float(face[2] * face[3]),
        )

    def _align_face(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        return self.recognizer.alignCrop(image, face)

    def _extract_embeddings_from_aligned_face(self, aligned_face: np.ndarray) -> List[np.ndarray]:
        embeddings = []
        for variant in self._enhance_face_crop(aligned_face):
            embeddings.append(self.recognizer.feature(variant))
        return embeddings

    def _extract_embeddings_from_face(self, image: np.ndarray, face: np.ndarray) -> List[np.ndarray]:
        aligned_face = self._align_face(image, face)
        return self._extract_embeddings_from_aligned_face(aligned_face)

    def analyze_face(self, image_source) -> FaceAnalysisResult:
        image = self._load_image(image_source)
        source = image_source if isinstance(image_source, str) else None
        all_embeddings: List[np.ndarray] = []
        best_face = None
        best_metrics = None

        for variant in self._generate_detection_variants(image):
            face = self._detect_largest_face(variant)
            if face is None:
                continue

            aligned_face = self._align_face(variant, face)
            all_embeddings.extend(self._extract_embeddings_from_aligned_face(aligned_face))
            metrics = self._calculate_face_quality(variant, face, aligned_face)

            if best_metrics is None or self._analysis_priority(metrics, face) > self._analysis_priority(
                best_metrics, best_face
            ):
                best_face = face.copy()
                best_metrics = metrics

        if best_metrics is None or best_face is None:
            return FaceAnalysisResult(
                source=source,
                image=image,
                face=None,
                embeddings=[],
                detection_confidence=None,
                face_area_ratio=None,
                sharpness=None,
                brightness=None,
                center_score=None,
                quality_score=None,
            )

        return FaceAnalysisResult(
            source=source,
            image=image,
            face=best_face,
            embeddings=all_embeddings,
            detection_confidence=best_metrics["detection_confidence"],
            face_area_ratio=best_metrics["face_area_ratio"],
            sharpness=best_metrics["sharpness"],
            brightness=best_metrics["brightness"],
            center_score=best_metrics["center_score"],
            quality_score=best_metrics["quality_score"],
        )

    def detect_largest_face(self, image_source):
        analysis = self.analyze_face(image_source)
        return analysis.image, analysis.face

    def extract_embedding(self, image_source) -> Optional[np.ndarray]:
        embeddings = self.analyze_face(image_source).embeddings
        return embeddings[0] if embeddings else None

    def extract_embeddings(self, image_source) -> List[np.ndarray]:
        return self.analyze_face(image_source).embeddings

    def save_detection_preview(self, image_source, output_path: str, analysis=None) -> bool:
        face_analysis = analysis or self.analyze_face(image_source)
        preview = face_analysis.image.copy()

        if face_analysis.face is not None:
            x, y, width, height = map(int, face_analysis.face[:4])
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                preview,
                "face",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, preview)
        return face_analysis.face is not None

    def cosine_similarity(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        return float(
            self.recognizer.match(
                feature_a, feature_b, cv2.FaceRecognizerSF_FR_COSINE
            )
        )

    def is_match(self, similarity: float) -> bool:
        return similarity >= self.score_threshold


class UserFaceDatabase:
    FIELDNAMES = ["user_id", "registered_plate", "face_dir"]
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    MAX_SELECTED_QUERY_ANALYSES = 3
    MAX_SELECTED_QUERY_EMBEDDINGS = 6
    MAX_SELECTED_RECORD_EMBEDDINGS = 4
    VIDEO_SAMPLE_FRAME_COUNT = 12
    MAX_VIDEO_FACE_RECORDS = 6

    def __init__(self, csv_path: str, face_recognizer: FaceRecognitionService):
        self.csv_path = csv_path
        self.face_recognizer = face_recognizer
        self.faces_root = os.path.join(os.path.dirname(self.csv_path), "faces")
        self.records = self._load_records()

    def _resolve_face_image_path(self, raw_path: str) -> str:
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.abspath(os.path.join(os.path.dirname(self.csv_path), raw_path))

    def _resolve_face_dir(self, raw_path: str) -> str:
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.abspath(os.path.join(os.path.dirname(self.csv_path), raw_path))

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        safe = "".join(char for char in user_id if char.isalnum() or char in ("-", "_"))
        return safe or "user"

    @staticmethod
    def _relative_to_csv_dir(csv_path: str, target_path: str) -> str:
        return os.path.relpath(target_path, os.path.dirname(csv_path)).replace("\\", "/")

    def _list_face_sources(self, face_dir: str) -> List[str]:
        if not os.path.isdir(face_dir):
            return []
        source_paths = []
        for entry in sorted(os.listdir(face_dir)):
            path = os.path.join(face_dir, entry)
            if os.path.isdir(path):
                continue
            extension = os.path.splitext(entry)[1].lower()
            if extension in self.IMAGE_EXTENSIONS or extension in self.VIDEO_EXTENSIONS:
                source_paths.append(path)
        return source_paths

    @staticmethod
    def _build_video_frame_label(video_path: str, frame_index: int) -> str:
        return f"{video_path}#frame={frame_index}"

    def _sample_video_frames(self, video_path: str) -> List[Tuple[int, np.ndarray]]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return []

        frames: List[Tuple[int, np.ndarray]] = []
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                sample_count = min(self.VIDEO_SAMPLE_FRAME_COUNT, frame_count)
                frame_indices = np.linspace(
                    0, max(frame_count - 1, 0), num=sample_count, dtype=int
                ).tolist()
                frame_indices = sorted(set(int(index) for index in frame_indices))

                for frame_index in frame_indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, frame = capture.read()
                    if success and frame is not None:
                        frames.append((frame_index, frame))
                return frames

            frame_index = 0
            while len(frames) < self.VIDEO_SAMPLE_FRAME_COUNT:
                success, frame = capture.read()
                if not success or frame is None:
                    break
                frames.append((frame_index, frame))
                frame_index += 1
        finally:
            capture.release()

        return frames

    def _build_face_records_for_image(
        self,
        user_id: str,
        registered_plate: str,
        face_dir: str,
        face_image_path: str,
    ) -> List[UserFaceRecord]:
        embeddings = self.face_recognizer.extract_embeddings(face_image_path)
        if not embeddings:
            return []
        return [
            UserFaceRecord(
                user_id=user_id,
                registered_plate=registered_plate,
                face_image_path=face_image_path,
                face_dir=face_dir,
                embeddings=embeddings,
            )
        ]

    def _build_face_records_for_video(
        self,
        user_id: str,
        registered_plate: str,
        face_dir: str,
        video_path: str,
    ) -> List[UserFaceRecord]:
        frame_analyses = []
        for frame_index, frame in self._sample_video_frames(video_path):
            analysis = self.face_recognizer.analyze_face(frame)
            if analysis.face is None or not analysis.embeddings:
                continue
            frame_analyses.append((analysis.quality_score or 0.0, frame_index, analysis))

        selected_analyses = sorted(
            frame_analyses, key=lambda item: (item[0], item[1]), reverse=True
        )[: self.MAX_VIDEO_FACE_RECORDS]

        return [
            UserFaceRecord(
                user_id=user_id,
                registered_plate=registered_plate,
                face_image_path=self._build_video_frame_label(video_path, frame_index),
                face_dir=face_dir,
                embeddings=analysis.embeddings,
            )
            for _, frame_index, analysis in selected_analyses
        ]

    def _build_face_records_for_source(
        self,
        user_id: str,
        registered_plate: str,
        face_dir: str,
        source_path: str,
    ) -> List[UserFaceRecord]:
        extension = os.path.splitext(source_path)[1].lower()
        if extension in self.IMAGE_EXTENSIONS:
            return self._build_face_records_for_image(
                user_id, registered_plate, face_dir, source_path
            )
        if extension in self.VIDEO_EXTENSIONS:
            return self._build_face_records_for_video(
                user_id, registered_plate, face_dir, source_path
            )
        return []

    def _default_face_dir_for_user(self, user_id: str) -> str:
        return os.path.join(self.faces_root, self._safe_user_id(user_id))

    def _read_rows_without_validation(self) -> List[dict]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"User database '{self.csv_path}' not found.")

        with open(self.csv_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            return list(reader)

    def _migrate_if_needed(self):
        rows = self._read_rows_without_validation()
        if not rows:
            fieldnames = self.FIELDNAMES
            existing_header = []
            with open(self.csv_path, "r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                existing_header = reader.fieldnames or []
            if existing_header == fieldnames:
                return
            with open(self.csv_path, "w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
            return

        with open(self.csv_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []

        requires_migration = set(fieldnames) != set(self.FIELDNAMES)
        migrated_rows = []

        for row in rows:
            user_id = row.get("user_id", "").strip()
            if not user_id:
                continue

            registered_plate = row.get("registered_plate", "").strip()

            raw_face_dir = row.get("face_dir", "").strip()
            raw_face_image_path = row.get("face_image_path", "").strip()

            if raw_face_dir:
                absolute_face_dir = self._resolve_face_dir(raw_face_dir)
            else:
                absolute_face_dir = self._default_face_dir_for_user(user_id)
                requires_migration = True

            if raw_face_image_path:
                absolute_face_image = self._resolve_face_image_path(raw_face_image_path)
                if os.path.exists(absolute_face_image):
                    os.makedirs(absolute_face_dir, exist_ok=True)
                    extension = os.path.splitext(absolute_face_image)[1].lower() or ".jpg"
                    destination_path = os.path.join(absolute_face_dir, f"1{extension}")
                    if os.path.abspath(absolute_face_image) != os.path.abspath(destination_path):
                        shutil.copy2(absolute_face_image, destination_path)
                    requires_migration = True

            os.makedirs(absolute_face_dir, exist_ok=True)
            migrated_rows.append(
                {
                    "user_id": user_id,
                    "registered_plate": registered_plate,
                    "face_dir": self._relative_to_csv_dir(self.csv_path, absolute_face_dir),
                }
            )

        if not requires_migration:
            return

        with open(self.csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(migrated_rows)

    def _load_records(self) -> List[UserFaceRecord]:
        self._migrate_if_needed()

        records: List[UserFaceRecord] = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = set(self.FIELDNAMES)
            missing = required_columns.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"User database is missing required columns: {', '.join(sorted(missing))}"
                )

            for row in reader:
                user_id = row["user_id"].strip()
                registered_plate = row["registered_plate"].strip()
                face_dir = self._resolve_face_dir(row["face_dir"].strip())
                face_sources = self._list_face_sources(face_dir)
                for source_path in face_sources:
                    records.extend(
                        self._build_face_records_for_source(
                            user_id=user_id,
                            registered_plate=registered_plate,
                            face_dir=face_dir,
                            source_path=source_path,
                        )
                    )
        return records

    def _build_records_by_user(self, candidate_user_ids: Optional[set] = None) -> dict:
        records_by_user = {}
        for record in self.records:
            if candidate_user_ids is not None and record.user_id not in candidate_user_ids:
                continue
            user_entries = records_by_user.setdefault(record.user_id, [])
            for embedding in record.embeddings:
                user_entries.append((record, embedding))
        return records_by_user

    def find_user_by_plate(self, scanned_plate: str) -> Optional[dict]:
        normalized_scanned_plate = OCRRecognizerBase.normalize_plate_for_compare(scanned_plate)
        users_by_id = {}
        for record in self.records:
            users_by_id.setdefault(record.user_id, record)

        for record in users_by_id.values():
            normalized_registered_plate = OCRRecognizerBase.normalize_plate_for_compare(
                record.registered_plate
            )
            if normalized_registered_plate == normalized_scanned_plate:
                return {
                    "user_id": record.user_id,
                    "registered_plate": record.registered_plate,
                    "face_image_path": record.face_image_path,
                    "face_dir": record.face_dir,
                }
        return None

    @staticmethod
    def _build_weighted_embedding_prototype(
        embeddings: List[np.ndarray], similarities: List[float]
    ) -> np.ndarray:
        weights = np.array([max(float(similarity), 0.0) for similarity in similarities])
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones(len(embeddings), dtype=np.float32)

        stacked = np.vstack([embedding.reshape(-1) for embedding in embeddings]).astype(
            np.float32
        )
        prototype = np.average(stacked, axis=0, weights=weights.astype(np.float32))
        norm = np.linalg.norm(prototype)
        if np.isclose(norm, 0.0):
            prototype = stacked[0]
            norm = np.linalg.norm(prototype)
        prototype = prototype / max(norm, 1e-12)
        return prototype.reshape(embeddings[0].shape).astype(np.float32)

    def _score_user_match(
        self, query_analyses: List[FaceAnalysisResult], user_entries: List[tuple]
    ) -> Optional[dict]:
        if not query_analyses or not user_entries:
            return None

        query_entries = []
        for analysis_index, analysis in enumerate(query_analyses):
            for query_embedding in analysis.embeddings:
                query_entries.append((analysis_index, analysis, query_embedding))

        if not query_entries:
            return None

        pairwise_scores = []
        for query_index, (analysis_index, analysis, query_embedding) in enumerate(query_entries):
            for entry_index, (record, record_embedding) in enumerate(user_entries):
                score = self.face_recognizer.cosine_similarity(
                    query_embedding, record_embedding
                )
                pairwise_scores.append(
                    (score, query_index, analysis_index, analysis, entry_index, record)
                )

        best_score, best_query_index, _, _, best_entry_index, best_record = max(
            pairwise_scores, key=lambda item: item[0]
        )
        best_record_embedding = user_entries[best_entry_index][1]
        best_query_embedding = query_entries[best_query_index][2]

        analyses_by_score = []
        for analysis_index, analysis in enumerate(query_analyses):
            analysis_best_score = max(
                (
                    self.face_recognizer.cosine_similarity(query_embedding, record_embedding)
                    for entry_analysis_index, _, query_embedding in query_entries
                    if entry_analysis_index == analysis_index
                    for _, record_embedding in user_entries
                ),
                default=-1.0,
            )
            if analysis_best_score < 0.0:
                continue
            analyses_by_score.append(
                (
                    analysis_best_score,
                    analysis.quality_score or 0.0,
                    analysis_index,
                    analysis,
                )
            )

        selected_analyses = [
            analysis
            for _, _, _, analysis in sorted(
                analyses_by_score, key=lambda item: (item[0], item[1]), reverse=True
            )[: self.MAX_SELECTED_QUERY_ANALYSES]
        ]
        if not selected_analyses:
            selected_analyses = [query_entries[best_query_index][1]]

        top_query_matches = []
        for analysis in selected_analyses:
            best_embeddings_for_analysis = sorted(
                (
                    (
                        max(
                            self.face_recognizer.cosine_similarity(
                                query_embedding, record_embedding
                            )
                            for _, record_embedding in user_entries
                        ),
                        query_embedding,
                        analysis.quality_score or 0.0,
                    )
                    for query_embedding in analysis.embeddings
                ),
                key=lambda item: (item[0], item[2]),
                reverse=True,
            )[:2]
            top_query_matches.extend(best_embeddings_for_analysis)

        top_query_matches = sorted(
            top_query_matches, key=lambda item: (item[0], item[2]), reverse=True
        )[: self.MAX_SELECTED_QUERY_EMBEDDINGS]
        if not top_query_matches:
            top_query_matches = [(best_score, best_query_embedding, 1.0)]

        selected_query_embeddings = [embedding for _, embedding, _ in top_query_matches]
        selected_query_weights = [
            max(similarity, 0.0) * max(quality_score, 0.1)
            for similarity, _, quality_score in top_query_matches
        ]

        top_record_matches = sorted(
            (
                (
                    max(
                        self.face_recognizer.cosine_similarity(
                            query_embedding, record_embedding
                        )
                        for query_embedding in selected_query_embeddings
                    ),
                    record_embedding,
                )
                for _, record_embedding in user_entries
            ),
            key=lambda item: item[0],
            reverse=True,
        )[: self.MAX_SELECTED_RECORD_EMBEDDINGS]
        if not top_record_matches:
            top_record_matches = [(best_score, best_record_embedding)]

        query_prototype = self._build_weighted_embedding_prototype(
            selected_query_embeddings,
            selected_query_weights,
        )
        record_prototype = self._build_weighted_embedding_prototype(
            [embedding for _, embedding in top_record_matches],
            [similarity for similarity, _ in top_record_matches],
        )
        prototype_score = self.face_recognizer.cosine_similarity(
            query_prototype, record_prototype
        )

        return {
            "record": best_record,
            "similarity": max(best_score, prototype_score),
            "selected_sources": [analysis.source for analysis in selected_analyses if analysis.source],
        }

    def find_best_match(self, image_source):
        return self.find_best_match_from_sources([image_source])

    def find_best_match_from_sources(self, image_sources: List[str]):
        face_analyses = [
            self.face_recognizer.analyze_face(image_source) for image_source in image_sources
        ]
        return self.find_best_match_from_analyses(face_analyses)

    def _find_best_match_from_usable_analyses(
        self,
        usable_analyses: List[FaceAnalysisResult],
        candidate_user_ids: Optional[set] = None,
    ) -> Optional[dict]:
        if not usable_analyses:
            return None

        records_by_user = self._build_records_by_user(candidate_user_ids)
        if not records_by_user:
            return None

        best_record = None
        best_score = -1.0
        best_selected_sources = []
        for user_entries in records_by_user.values():
            user_match = self._score_user_match(usable_analyses, user_entries)
            if user_match is None:
                continue
            if user_match["similarity"] > best_score:
                best_record = user_match["record"]
                best_score = user_match["similarity"]
                best_selected_sources = user_match.get("selected_sources", [])

        if best_record is None or not self.face_recognizer.is_match(best_score):
            return None

        return {
            "user_id": best_record.user_id,
            "registered_plate": best_record.registered_plate,
            "face_image_path": best_record.face_image_path,
            "face_dir": best_record.face_dir,
            "similarity": best_score,
            "selected_face_sources": best_selected_sources,
        }

    def find_best_match_from_analyses(self, face_analyses: List[FaceAnalysisResult]):
        usable_analyses = [
            analysis
            for analysis in face_analyses
            if analysis.face is not None and analysis.embeddings
        ]
        if not usable_analyses:
            return None

        best_multi_match = self._find_best_match_from_usable_analyses(usable_analyses)
        if len(usable_analyses) == 1:
            return best_multi_match

        best_single_match = None
        for analysis in usable_analyses:
            single_match = self._find_best_match_from_usable_analyses([analysis])
            if single_match is None:
                continue
            if (
                best_single_match is None
                or single_match["similarity"] > best_single_match["similarity"]
            ):
                best_single_match = single_match

        if best_single_match is not None and (
            best_multi_match is None
            or best_single_match["similarity"] > best_multi_match["similarity"]
        ):
            return best_single_match

        return best_multi_match

    def find_best_match_for_user_from_analyses(
        self, face_analyses: List[FaceAnalysisResult], user_id: str
    ) -> Optional[dict]:
        usable_analyses = [
            analysis
            for analysis in face_analyses
            if analysis.face is not None and analysis.embeddings
        ]
        if not usable_analyses:
            return None

        best_multi_match = self._find_best_match_from_usable_analyses(
            usable_analyses, candidate_user_ids={user_id}
        )
        if len(usable_analyses) == 1:
            return best_multi_match

        best_single_match = None
        for analysis in usable_analyses:
            single_match = self._find_best_match_from_usable_analyses(
                [analysis], candidate_user_ids={user_id}
            )
            if single_match is None:
                continue
            if (
                best_single_match is None
                or single_match["similarity"] > best_single_match["similarity"]
            ):
                best_single_match = single_match

        if best_single_match is not None and (
            best_multi_match is None
            or best_single_match["similarity"] > best_multi_match["similarity"]
        ):
            return best_single_match

        return best_multi_match

    def source_has_detectable_face(self, source_path: str) -> bool:
        preview_records = self._build_face_records_for_source(
            user_id="preview",
            registered_plate="",
            face_dir="",
            source_path=source_path,
        )
        return len(preview_records) > 0

    def upsert_user(
        self,
        user_id: str,
        registered_plate: str,
        face_dir: str,
    ):
        rows = []
        fieldnames = self.FIELDNAMES

        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    rows.append(
                        {
                            "user_id": row.get("user_id", "").strip(),
                            "registered_plate": row.get("registered_plate", "").strip(),
                            "face_dir": row.get("face_dir", "").strip(),
                        }
                    )

        updated = False
        for row in rows:
            if row["user_id"] == user_id:
                row["registered_plate"] = registered_plate
                row["face_dir"] = face_dir
                updated = True
                break

        if not updated:
            rows.append(
                {
                    "user_id": user_id,
                    "registered_plate": registered_plate,
                    "face_dir": face_dir,
                }
            )

        with open(self.csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        self.records = self._load_records()

    @staticmethod
    def copy_face_source(source_path: str, destination_path: str):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)

    @staticmethod
    def copy_face_image(source_image_path: str, destination_image_path: str):
        UserFaceDatabase.copy_face_source(source_image_path, destination_image_path)
