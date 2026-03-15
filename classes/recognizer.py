import csv
import math
import os
import re
import traceback
import tempfile
from typing import List, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
import yaml
from yolov5.models.yolo import DetectionModel
from yolov5.utils.general import non_max_suppression


class VietnamesePlatePostProcessor:
    @staticmethod
    def _replace_by_position(
        value: str,
        probs: Union[npt.NDArray, None],
        mapping: dict,
        threshold: float,
        force: bool = False,
    ) -> str:
        corrected = []
        for index, char in enumerate(value):
            if char not in mapping:
                corrected.append(char)
                continue

            if force:
                corrected.append(mapping[char])
                continue

            if probs is not None and index < len(probs) and float(probs[index]) < threshold:
                corrected.append(mapping[char])
            else:
                corrected.append(char)
        return "".join(corrected)

    @classmethod
    def correct_vietnamese_plate_confusions(
        cls, plate_text: str, char_probabilities: Union[npt.NDArray, None] = None
    ) -> str:
        cleaned = re.sub(r"[^A-Z0-9]", "", plate_text.upper())
        if len(cleaned) < 7:
            return cleaned

        digit_slot_map = {
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "T": "7",
            "B": "8",
        }
        first_series_letter_map = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "6": "G",
            "8": "B",
        }
        second_series_digit_map = {
            "O": "0",
            "Q": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "T": "7",
        }
        dg_confusion_map = {"D": "G", "G": "D"}

        if len(cleaned) == 8 and cleaned[:2].isdigit() and cleaned[-5:].isdigit():
            series = cleaned[2:3]
            serial = cleaned[3:]
            return f"{cleaned[:2]}-{series} {serial[:3]}.{serial[3:]}"

        if len(cleaned) not in (8, 9):
            return cleaned

        province = cls._replace_by_position(
            cleaned[:2],
            char_probabilities[:2] if char_probabilities is not None else None,
            digit_slot_map,
            1.0,
            force=True,
        )

        if len(cleaned) == 8:
            series = cleaned[2:3]
            serial = cleaned[3:]
            series_probs = char_probabilities[2:3] if char_probabilities is not None else None
            serial_probs = char_probabilities[3:8] if char_probabilities is not None else None
        else:
            series = cleaned[2:4]
            serial = cleaned[4:]
            series_probs = char_probabilities[2:4] if char_probabilities is not None else None
            serial_probs = char_probabilities[4:9] if char_probabilities is not None else None

        first_series = cls._replace_by_position(
            series[:1],
            series_probs[:1] if series_probs is not None else None,
            first_series_letter_map,
            1.0,
            force=True,
        )

        second_series = series[1:]
        if second_series:
            second_series = cls._replace_by_position(
                second_series,
                series_probs[1:] if series_probs is not None else None,
                second_series_digit_map,
                0.92,
            )
            second_series = cls._replace_by_position(
                second_series,
                series_probs[1:] if series_probs is not None else None,
                dg_confusion_map,
                0.85,
            )

        serial = cls._replace_by_position(
            serial,
            serial_probs,
            digit_slot_map,
            1.0,
            force=True,
        )

        if (
            serial_probs is not None
            and len(serial) >= 2
            and serial[-1] == serial[-2]
            and serial[-1] in {"2", "7"}
            and float(serial_probs[-2]) < 0.90
        ):
            swapped = "7" if serial[-2] == "2" else "2"
            serial = f"{serial[:-2]}{swapped}{serial[-1]}"

        series = f"{first_series}{second_series}"
        return f"{province}-{series} {serial[:3]}.{serial[3:]}"

    @classmethod
    def normalize_vietnamese_plate(
        cls, plate_text: str, char_probabilities: Union[npt.NDArray, None] = None
    ) -> str:
        cleaned = re.sub(r"[^A-Z0-9]", "", plate_text.upper())
        if len(cleaned) < 7:
            return cleaned
        return cls.correct_vietnamese_plate_confusions(cleaned, char_probabilities)

    @staticmethod
    def calculate_plate_confidence(
        plate_text: str, char_probabilities: npt.NDArray, max_plate_slots: int
    ) -> float:
        effective_length = min(len(plate_text), max_plate_slots)
        if effective_length == 0:
            return 0.0
        return float(np.mean(char_probabilities[:effective_length]))


class OCRRecognizerBase(VietnamesePlatePostProcessor):
    @staticmethod
    def normalize_plate_for_compare(plate_text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", plate_text.upper())

    def process_cropped_images(self, cropped_images_dir: str, results_dir: str):
        if not os.path.exists(cropped_images_dir):
            raise FileNotFoundError(f"Directory '{cropped_images_dir}' does not exist.")

        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "ocr_results.csv")

        with open(results_file, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Image Name", "Extracted Value", "Confidence"])

            for image_name in os.listdir(cropped_images_dir):
                image_path = os.path.join(cropped_images_dir, image_name)
                if os.path.isfile(image_path) and image_path.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    raw_value, confidences = self.run(image_path, return_confidence=True)
                    extracted_value = (
                        self.normalize_vietnamese_plate(raw_value, confidences)
                        if raw_value != "N/A"
                        else raw_value
                    )
                    confidence = ""

                    if raw_value != "N/A" and confidences is not None and len(confidences) > 0:
                        confidence = (
                            f"{self.calculate_plate_confidence(raw_value, confidences, len(confidences)):.4f}"
                        )

                    writer.writerow(
                        [image_name.replace("_cropped", ""), extracted_value, confidence]
                    )

        print(f"Results saved to {results_file}")


class ONNXPlateRecognizer(OCRRecognizerBase):
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config = self.load_config(config_path)
        self.model = self.load_model(model_path)

    @staticmethod
    def load_config(config_path: str) -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file '{config_path}' not found.")
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)

    @staticmethod
    def load_model(model_path: str) -> ort.InferenceSession:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        return ort.InferenceSession(model_path, providers=providers)

    @staticmethod
    def read_plate_image(image_path: str) -> npt.NDArray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File '{image_path}' not found.")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image '{image_path}'.")
        return image

    @staticmethod
    def _normalize_array_image(image: npt.NDArray) -> npt.NDArray:
        image = np.asarray(image).squeeze()
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("Array input must have shape (H, W), (H, W, 1), or (H, W, 3).")

    def _load_image_from_source(
        self, source: Union[str, List[str], npt.NDArray, List[npt.NDArray]]
    ) -> Union[npt.NDArray, List[npt.NDArray]]:
        if isinstance(source, str):
            return self.read_plate_image(source)

        if isinstance(source, list):
            if all(isinstance(item, str) for item in source):
                return [self.read_plate_image(item) for item in source]
            if all(isinstance(item, np.ndarray) for item in source):
                return [self._normalize_array_image(item) for item in source]
            raise ValueError("List input must contain only strings or numpy arrays.")

        if isinstance(source, np.ndarray):
            return self._normalize_array_image(source)

        raise TypeError("Unsupported input type. Provide a path or numpy array.")

    def preprocess_image(
        self, image: npt.NDArray, img_height: int, img_width: int
    ) -> npt.NDArray:
        if isinstance(image, np.ndarray):
            image = [image]

        images = np.array(
            [
                cv2.resize(img.squeeze(), (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                for img in image
            ]
        )
        images = np.expand_dims(images, axis=-1)
        return images.astype(np.uint8)

    def postprocess_output(
        self,
        model_output: npt.NDArray,
        max_plate_slots: int,
        model_alphabet: str,
        pad_char: str,
        return_confidence: bool = False,
    ) -> Union[str, Tuple[str, npt.NDArray]]:
        predictions = model_output.reshape((-1, max_plate_slots, len(model_alphabet)))
        prediction_indices = np.argmax(predictions, axis=-1)
        alphabet_array = np.array(list(model_alphabet))
        plate_chars = alphabet_array[prediction_indices]
        plates = ["".join(plate).replace(pad_char, "") for plate in plate_chars]

        if return_confidence:
            probabilities = np.max(predictions, axis=-1)
            return plates[0] if plates else "N/A", probabilities[0] if len(probabilities) > 0 else None
        return plates[0] if plates else "N/A"

    def run(
        self,
        source: Union[str, List[str], npt.NDArray, List[npt.NDArray]],
        return_confidence: bool = False,
    ) -> Union[str, Tuple[str, npt.NDArray]]:
        try:
            inputs = self._load_image_from_source(source)
            inputs = self.preprocess_image(
                inputs, self.config["img_height"], self.config["img_width"]
            )
            outputs: List[npt.NDArray] = self.model.run(None, {"input": inputs})
            return self.postprocess_output(
                outputs[0],
                self.config["max_plate_slots"],
                self.config["alphabet"],
                self.config["pad_char"],
                return_confidence=return_confidence,
            )
        except Exception as exc:
            print(f"Error during OCR processing: {exc}")
            traceback.print_exc()
            return ("N/A", None) if return_confidence else "N/A"


class VietnameseYOLOPlateRecognizer(OCRRecognizerBase):
    def __init__(self, model_path: str, conf: float = 0.35, iou: float = 0.45, max_det: int = 16):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.input_size = 640
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.names = self.load_model(model_path)

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = DetectionModel(checkpoint["model"].yaml)
        model.load_state_dict(checkpoint["model"].state_dict())
        model.to(self.device).eval()
        names = list(getattr(checkpoint["model"], "names", []))
        return model, names

    @staticmethod
    def _load_image(image_path: str) -> npt.NDArray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image '{image_path}'.")
        return image

    @staticmethod
    def _letterbox(image: npt.NDArray, size: int, color=(114, 114, 114)):
        height, width = image.shape[:2]
        scale = min(size / height, size / width)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        pad_w = size - resized_width
        pad_h = size - resized_height
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        bordered = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return bordered, scale, (left, top)

    @staticmethod
    def _scale_boxes(boxes: torch.Tensor, scale: float, pad: Tuple[int, int], image_shape):
        left, top = pad
        boxes = boxes.clone()
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes[:, :4] /= scale
        height, width = image_shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        return boxes

    @staticmethod
    def _change_contrast(image: npt.NDArray) -> npt.NDArray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        merged = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _rotate_image(image: npt.NDArray, angle: float) -> npt.NDArray:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rotation = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(
            image, rotation, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )

    def _compute_skew(self, image: npt.NDArray, center_threshold: int) -> float:
        height, width = image.shape[:2]
        blurred = cv2.medianBlur(image, 3)
        edges = cv2.Canny(
            blurred, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True
        )
        lines = cv2.HoughLinesP(
            edges,
            1,
            math.pi / 180,
            30,
            minLineLength=width / 1.5,
            maxLineGap=height / 3.0,
        )
        if lines is None:
            return 0.0

        min_line = 100
        min_line_pos = 0
        for index in range(len(lines)):
            for x1, y1, x2, y2 in lines[index]:
                center_point = [((x1 + x2) / 2), ((y1 + y2) / 2)]
                if center_threshold == 1 and center_point[1] < 7:
                    continue
                if center_point[1] < min_line:
                    min_line = center_point[1]
                    min_line_pos = index

        angle = 0.0
        count = 0
        for x1, y1, x2, y2 in lines[min_line_pos]:
            current_angle = np.arctan2(y2 - y1, x2 - x1)
            if math.fabs(current_angle) <= 30:
                angle += current_angle
                count += 1
        if count == 0:
            return 0.0
        return (angle / count) * 180 / math.pi

    def _deskew(self, image: npt.NDArray, change_contrast: int, center_threshold: int):
        source = self._change_contrast(image) if change_contrast == 1 else image
        return self._rotate_image(source, self._compute_skew(source, center_threshold))

    def _predict_characters(self, image: npt.NDArray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded, scale, pad = self._letterbox(rgb, self.input_size)
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(tensor)[0]

        prediction = non_max_suppression(
            prediction, self.conf, self.iou, max_det=self.max_det
        )[0]
        if prediction.shape[0] == 0:
            return []

        prediction = self._scale_boxes(prediction, scale, pad, image.shape)
        detections = []
        for row in prediction:
            x1, y1, x2, y2, confidence, class_id = row.tolist()
            detections.append(
                {
                    "x_center": (x1 + x2) / 2,
                    "y_center": (y1 + y2) / 2,
                    "label": self.names[int(class_id)],
                    "confidence": float(confidence),
                }
            )
        return detections

    @staticmethod
    def _is_two_line_plate(detections: List[dict]) -> bool:
        if len(detections) < 7:
            return False
        y_values = [item["y_center"] for item in detections]
        return max(y_values) - min(y_values) > 18

    def _decode_plate(self, detections: List[dict]) -> Tuple[str, Union[npt.NDArray, None]]:
        if len(detections) < 7 or len(detections) > 10:
            return "N/A", None

        if self._is_two_line_plate(detections):
            y_mean = sum(item["y_center"] for item in detections) / len(detections)
            line_1 = sorted(
                [item for item in detections if item["y_center"] <= y_mean],
                key=lambda item: item["x_center"],
            )
            line_2 = sorted(
                [item for item in detections if item["y_center"] > y_mean],
                key=lambda item: item["x_center"],
            )
            ordered = line_1 + line_2
        else:
            ordered = sorted(detections, key=lambda item: item["x_center"])

        plate = "".join(item["label"] for item in ordered)
        confidences = np.array([item["confidence"] for item in ordered], dtype=np.float32)
        return plate, confidences

    def run(self, source: Union[str, npt.NDArray], return_confidence: bool = False):
        try:
            if isinstance(source, str):
                image = self._load_image(source)
            elif isinstance(source, np.ndarray):
                image = source
            else:
                raise TypeError("Vietnamese YOLO OCR expects a file path or image array.")
            candidates = [image]
            for change_contrast in range(2):
                for center_threshold in range(2):
                    candidates.append(
                        self._deskew(image, change_contrast, center_threshold)
                    )

            best_plate = "N/A"
            best_confidences = None
            best_score = -1.0

            for candidate in candidates:
                detections = self._predict_characters(candidate)
                plate, confidences = self._decode_plate(detections)
                if plate == "N/A" or confidences is None:
                    continue

                score = float(np.mean(confidences))
                if score > best_score:
                    best_plate = plate
                    best_confidences = confidences
                    best_score = score

            if return_confidence:
                return best_plate, best_confidences
            return best_plate

        except Exception as exc:
            print(f"Error during Vietnamese OCR processing: {exc}")
            traceback.print_exc()
            return ("N/A", None) if return_confidence else "N/A"


def create_plate_recognizer(model_path: str, config_path: Union[str, None] = None):
    if model_path.lower().endswith(".pt"):
        return VietnameseYOLOPlateRecognizer(model_path)
    if config_path is None:
        raise ValueError("config_path is required for ONNX OCR models.")
    return ONNXPlateRecognizer(model_path, config_path)
