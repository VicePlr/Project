import os
import traceback

import cv2
import torch
from yolov5.models.yolo import DetectionModel


class YOLOv5Inference:
    def __init__(
        self,
        model_path,
        conf=0.25,
        iou=0.45,
        agnostic=False,
        multi_label=False,
        max_det=3,
        input_size=640,
        margin_factor=0.08,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        self.model = DetectionModel(checkpoint["model"].yaml)
        self.model.load_state_dict(checkpoint["model"].state_dict())
        self.model.to(self.device).eval()

        self.conf = conf
        self.iou = iou
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.input_size = input_size
        self.margin_factor = margin_factor

    @staticmethod
    def _load_image(img_url):
        if isinstance(img_url, str):
            img = cv2.imread(img_url, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {img_url}")
            return img
        if hasattr(img_url, "shape"):
            return img_url.copy()
        raise TypeError("Image source must be a file path or numpy array.")

    @staticmethod
    def _letterbox(image, size, color=(114, 114, 114)):
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
    def _scale_boxes(boxes, scale, pad, image_shape):
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
    def _expand_box(box, image_shape, margin_factor):
        x1, y1, x2, y2 = box
        height, width = image_shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1

        margin_x = int(box_width * margin_factor)
        margin_y = int(box_height * margin_factor)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        return x1, y1, x2, y2

    def infer(self, img_url, size=None):
        """Run inference for a single image."""
        size = size or self.input_size

        try:
            original_bgr = self._load_image(img_url)
            img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            img_padded, scale, pad = self._letterbox(img_rgb, size)

            img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.model(img_tensor)[0]

            prediction = self.non_max_suppression(
                prediction,
                conf_thres=self.conf,
                iou_thres=self.iou,
                agnostic=self.agnostic,
                multi_label=self.multi_label,
                max_det=self.max_det,
            )

            if prediction and len(prediction[0]) > 0:
                prediction[0] = self._scale_boxes(
                    prediction[0], scale, pad, original_bgr.shape
                )
                prediction[0] = prediction[0][prediction[0][:, 4].argsort(descending=True)]

            class Results:
                def __init__(self, pred, original_image):
                    self.pred = pred
                    self.ims = [original_image]

            return Results(prediction, original_bgr)

        except Exception as exc:
            print(f"Error while processing image '{img_url}': {exc}")
            traceback.print_exc()
            return None

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        max_det=300,
    ):
        """Apply non-max suppression to YOLO predictions."""
        from yolov5.utils.general import non_max_suppression as nms

        return nms(
            prediction,
            conf_thres,
            iou_thres,
            agnostic=agnostic,
            multi_label=multi_label,
            max_det=max_det,
        )

    def process_results(
        self,
        results,
        img_name,
        output_dir="results",
        margin_factor=None,
        detected_subdir="bbox",
        cropped_subdir="cropped",
    ):
        """Draw detections and save cropped plate regions."""
        if results is None:
            return

        predictions = results.pred[0]
        if predictions.shape[0] == 0:
            print("No detections found after NMS.")
            return

        margin_factor = self.margin_factor if margin_factor is None else margin_factor
        detected_dir = os.path.join(output_dir, detected_subdir)
        cropped_dir = os.path.join(output_dir, cropped_subdir)
        os.makedirs(detected_dir, exist_ok=True)
        os.makedirs(cropped_dir, exist_ok=True)

        original_img = results.ims[0].copy()
        detected_img = original_img.copy()
        stem, ext = os.path.splitext(img_name)

        for index, prediction in enumerate(predictions, start=1):
            x1, y1, x2, y2 = map(int, prediction[:4].tolist())
            x1, y1, x2, y2 = self._expand_box(
                (x1, y1, x2, y2), original_img.shape, margin_factor
            )

            cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            confidence = float(prediction[4].item())
            label = f"{confidence:.2f}"
            cv2.putText(
                detected_img,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cropped_image = original_img[y1:y2, x1:x2]
            suffix = "" if len(predictions) == 1 else f"_{index}"
            cropped_path = os.path.join(cropped_dir, f"{stem}_cropped{suffix}{ext}")
            cv2.imwrite(cropped_path, cropped_image)

        detected_path = os.path.join(detected_dir, f"{stem}_detected{ext}")
        cv2.imwrite(detected_path, detected_img)

    def extract_crops(self, results, margin_factor=None):
        """Return cropped detections in memory for downstream processing."""
        if results is None:
            return []

        predictions = results.pred[0]
        if predictions.shape[0] == 0:
            return []

        margin_factor = self.margin_factor if margin_factor is None else margin_factor
        original_img = results.ims[0].copy()
        crops = []

        for prediction in predictions:
            x1, y1, x2, y2 = map(int, prediction[:4].tolist())
            x1, y1, x2, y2 = self._expand_box(
                (x1, y1, x2, y2), original_img.shape, margin_factor
            )
            crops.append(
                {
                    "box": (x1, y1, x2, y2),
                    "confidence": float(prediction[4].item()),
                    "image": original_img[y1:y2, x1:x2].copy(),
                }
            )

        return crops

    def process_directory(self, input_dir, output_dir):
        """Process every supported image in a directory."""
        os.makedirs(output_dir, exist_ok=True)

        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            if os.path.isfile(img_path) and os.path.splitext(img_path)[1].lower() in [
                ".png",
                ".jpg",
                ".jpeg",
            ]:
                results = self.infer(img_path)
                self.process_results(results, img_name, output_dir)
