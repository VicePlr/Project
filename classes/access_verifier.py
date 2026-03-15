from dataclasses import asdict, dataclass
from typing import List, Optional

from classes.detector import YOLOv5Inference
from classes.face_recognizer import FaceAnalysisResult, FaceRecognitionService, UserFaceDatabase
from classes.recognizer import OCRRecognizerBase, create_plate_recognizer


@dataclass
class FaceMatchResult:
    matched: bool
    user_id: Optional[str]
    face_similarity: Optional[float]
    registered_plate: Optional[str]
    reason: str
    face_images_used: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class PlateScanResult:
    scanned_plate: Optional[str]
    scanned_plate_confidence: Optional[float]
    reason: str

    def to_dict(self):
        return asdict(self)


@dataclass
class AccessVerificationResult:
    checkpoint_id: Optional[str]
    success: bool
    user_id: Optional[str]
    face_similarity: Optional[float]
    registered_plate: Optional[str]
    scanned_plate: Optional[str]
    scanned_plate_confidence: Optional[float]
    plate_matches: bool
    reason: str

    def to_dict(self):
        return asdict(self)


class AccessVerifier:
    def __init__(
        self,
        plate_detector_model_path: str,
        plate_detector_input_size: int,
        plate_ocr_model_path: str,
        plate_ocr_config_path: Optional[str],
        face_detector_model_path: str,
        face_recognizer_model_path: str,
        users_csv_path: str,
    ):
        self.plate_detector = YOLOv5Inference(
            plate_detector_model_path, input_size=plate_detector_input_size
        )
        self.plate_recognizer = create_plate_recognizer(
            plate_ocr_model_path, plate_ocr_config_path
        )
        self.face_service = FaceRecognitionService(
            face_detector_model_path, face_recognizer_model_path
        )
        self.user_database = UserFaceDatabase(users_csv_path, self.face_service)

    def analyze_face_image(self, face_image_path: str) -> FaceAnalysisResult:
        return self.face_service.analyze_face(face_image_path)

    def lookup_registered_user_by_plate(self, scanned_plate: str) -> Optional[dict]:
        return self.user_database.find_user_by_plate(scanned_plate)

    def match_face_analyses(self, face_analyses: List[FaceAnalysisResult]) -> FaceMatchResult:
        face_match = self.user_database.find_best_match_from_analyses(face_analyses)
        if face_match is None:
            return FaceMatchResult(
                matched=False,
                user_id=None,
                face_similarity=None,
                registered_plate=None,
                reason="No matching user face found in the database.",
                face_images_used=None,
            )

        return FaceMatchResult(
            matched=True,
            user_id=face_match["user_id"],
            face_similarity=face_match["similarity"],
            registered_plate=face_match["registered_plate"],
            reason="Face matched a registered user.",
            face_images_used=face_match.get("selected_face_sources"),
        )

    def match_face_analyses_for_registered_user(
        self, face_analyses: List[FaceAnalysisResult], registered_user: dict
    ) -> FaceMatchResult:
        face_match = self.user_database.find_best_match_for_user_from_analyses(
            face_analyses, registered_user["user_id"]
        )
        if face_match is None:
            return FaceMatchResult(
                matched=False,
                user_id=registered_user["user_id"],
                face_similarity=None,
                registered_plate=registered_user["registered_plate"],
                reason="Face did not match the enrolled face folder for the scanned plate.",
                face_images_used=None,
            )

        return FaceMatchResult(
            matched=True,
            user_id=face_match["user_id"],
            face_similarity=face_match["similarity"],
            registered_plate=face_match["registered_plate"],
            reason="Face matched the enrolled face folder for the scanned plate.",
            face_images_used=face_match.get("selected_face_sources"),
        )

    def match_face(self, face_image_path: str) -> FaceMatchResult:
        face_analysis = self.analyze_face_image(face_image_path)
        return self.match_face_analyses([face_analysis])

    def match_faces(self, face_image_paths: List[str]) -> FaceMatchResult:
        face_analyses = [self.analyze_face_image(face_image_path) for face_image_path in face_image_paths]
        return self.match_face_analyses(face_analyses)

    def scan_plate(self, vehicle_image_path: str, detection_results=None) -> PlateScanResult:
        results = detection_results or self.plate_detector.infer(vehicle_image_path)
        crops = self.plate_detector.extract_crops(results)
        best_plate = None
        best_confidence = -1.0

        for crop in crops:
            raw_plate, char_confidences = self.plate_recognizer.run(
                crop["image"], return_confidence=True
            )
            if raw_plate == "N/A" or char_confidences is None or len(char_confidences) == 0:
                continue

            normalized_plate = self.plate_recognizer.normalize_vietnamese_plate(
                raw_plate, char_confidences
            )
            score = OCRRecognizerBase.calculate_plate_confidence(
                raw_plate, char_confidences, len(char_confidences)
            )

            if score > best_confidence:
                best_plate = normalized_plate
                best_confidence = score

        if best_plate is None:
            return PlateScanResult(
                scanned_plate=None,
                scanned_plate_confidence=None,
                reason="No license plate could be read from the vehicle image.",
            )

        return PlateScanResult(
            scanned_plate=best_plate,
            scanned_plate_confidence=best_confidence,
            reason="License plate read successfully.",
        )

    @staticmethod
    def compare_registered_and_scanned_plate(
        registered_plate: Optional[str], scanned_plate: Optional[str]
    ) -> bool:
        if not registered_plate or not scanned_plate:
            return False

        normalized_registered_plate = OCRRecognizerBase.normalize_plate_for_compare(
            registered_plate
        )
        normalized_scanned_plate = OCRRecognizerBase.normalize_plate_for_compare(
            scanned_plate
        )
        return normalized_registered_plate == normalized_scanned_plate

    def verify(
        self,
        face_image_path: str,
        vehicle_image_path: str,
        checkpoint_id: Optional[str] = None,
    ) -> AccessVerificationResult:
        plate_scan = self.scan_plate(vehicle_image_path)
        if plate_scan.scanned_plate is None:
            return AccessVerificationResult(
                checkpoint_id=checkpoint_id,
                success=False,
                user_id=None,
                face_similarity=None,
                registered_plate=None,
                scanned_plate=None,
                scanned_plate_confidence=None,
                plate_matches=False,
                reason=plate_scan.reason,
            )

        registered_user = self.lookup_registered_user_by_plate(plate_scan.scanned_plate)
        if registered_user is None:
            return AccessVerificationResult(
                checkpoint_id=checkpoint_id,
                success=False,
                user_id=None,
                face_similarity=None,
                registered_plate=None,
                scanned_plate=plate_scan.scanned_plate,
                scanned_plate_confidence=plate_scan.scanned_plate_confidence,
                plate_matches=False,
                reason="Scanned plate is not registered in the user database.",
            )

        face_analysis = self.analyze_face_image(face_image_path)
        face_match = self.match_face_analyses_for_registered_user(
            [face_analysis], registered_user
        )
        if not face_match.matched:
            return AccessVerificationResult(
                checkpoint_id=checkpoint_id,
                success=False,
                user_id=registered_user["user_id"],
                face_similarity=None,
                registered_plate=registered_user["registered_plate"],
                scanned_plate=plate_scan.scanned_plate,
                scanned_plate_confidence=plate_scan.scanned_plate_confidence,
                plate_matches=True,
                reason=face_match.reason,
            )

        return AccessVerificationResult(
            checkpoint_id=checkpoint_id,
            success=True,
            user_id=face_match.user_id,
            face_similarity=face_match.face_similarity,
            registered_plate=registered_user["registered_plate"],
            scanned_plate=plate_scan.scanned_plate,
            scanned_plate_confidence=plate_scan.scanned_plate_confidence,
            plate_matches=True,
            reason="Scanned plate matched a registered user and the face matched that user's face folder.",
        )
