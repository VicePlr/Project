import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


class SCRFDFaceDetector:
    def __init__(
        self,
        model_path: str,
        input_size=(640, 640),
        det_thresh: float = 0.35,
        nms_thresh: float = 0.4,
    ):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_size = input_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_mean = 127.5
        self.input_std = 128.0
        self.center_cache = {}

        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.use_kps = len(outputs) in (9, 15)
        if len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        else:
            raise ValueError(
                f"Unsupported SCRFD output layout with {len(outputs)} outputs for {model_path}."
            )

    @staticmethod
    def _distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance_to_kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        preds = []
        for index in range(0, distance.shape[1], 2):
            px = points[:, index % 2] + distance[:, index]
            py = points[:, index % 2 + 1] + distance[:, index + 1]
            preds.extend([px, py])
        return np.stack(preds, axis=-1)

    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key in self.center_cache:
            return self.center_cache[key]

        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
            np.float32
        )
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape(
            (-1, 2)
        )
        if len(self.center_cache) < 100:
            self.center_cache[key] = anchor_centers
        return anchor_centers

    def _forward(self, image: np.ndarray):
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outputs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for index, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outputs[index][0]
                bbox_preds = net_outputs[index + self.fmc][0] * stride
                kps_preds = net_outputs[index + self.fmc * 2][0] * stride
            else:
                scores = net_outputs[index]
                bbox_preds = net_outputs[index + self.fmc] * stride
                kps_preds = net_outputs[index + self.fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            anchor_centers = self._get_anchor_centers(height, width, stride)

            positive_indices = np.where(scores >= self.det_thresh)[0]
            if len(positive_indices) == 0:
                continue

            bboxes = self._distance_to_bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[positive_indices])
            bboxes_list.append(bboxes[positive_indices])

            if self.use_kps:
                kpss = self._distance_to_kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[positive_indices])

        return scores_list, bboxes_list, kpss_list

    def _nms(self, detections: np.ndarray):
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            current = order[0]
            keep.append(current)

            xx1 = np.maximum(x1[current], x1[order[1:]])
            yy1 = np.maximum(y1[current], y1[order[1:]])
            xx2 = np.minimum(x2[current], x2[order[1:]])
            yy2 = np.minimum(y2[current], y2[order[1:]])

            width = np.maximum(0.0, xx2 - xx1 + 1)
            height = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width * height
            overlap = intersection / (areas[current] + areas[order[1:]] - intersection)

            remaining = np.where(overlap <= self.nms_thresh)[0]
            order = order[remaining + 1]

        return keep

    @staticmethod
    def _to_opencv_face(bbox: np.ndarray, kps: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2, score = bbox.astype(np.float32)
        return np.array(
            [x1, y1, x2 - x1, y2 - y1, *kps.reshape(-1).tolist(), float(score)],
            dtype=np.float32,
        )

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        image_height, image_width = image.shape[:2]
        image_ratio = float(image_height) / image_width
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if image_ratio > model_ratio:
            resized_height = self.input_size[1]
            resized_width = int(resized_height / image_ratio)
        else:
            resized_width = self.input_size[0]
            resized_height = int(resized_width * image_ratio)

        detection_scale = float(resized_height) / image_height
        resized_image = cv2.resize(image, (resized_width, resized_height))
        detection_image = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        detection_image[:resized_height, :resized_width, :] = resized_image

        scores_list, bboxes_list, kpss_list = self._forward(detection_image)
        if not scores_list or not bboxes_list:
            return None

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / detection_scale
        pre_detections = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_detections = pre_detections[order, :]

        if self.use_kps:
            kpss = np.vstack(kpss_list) / detection_scale
            kpss = kpss[order, :, :]
        else:
            kpss = None

        keep = self._nms(pre_detections)
        detections = pre_detections[keep, :]
        if detections.shape[0] == 0 or kpss is None:
            return None

        kpss = kpss[keep, :, :]
        faces = [self._to_opencv_face(bbox, kps) for bbox, kps in zip(detections, kpss)]
        return np.stack(faces).astype(np.float32)


def create_face_detector(model_path: str):
    model_name = os.path.basename(model_path).lower()
    if "det_10g" not in model_name and "scrfd" not in model_name:
        raise ValueError(
            f"Face detector must be an SCRFD model, got '{model_path}'."
        )
    return SCRFDFaceDetector(model_path)
