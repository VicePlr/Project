import logging
from contextlib import contextmanager


YOLOV5_LOGGER_NAME = "yolov5"


@contextmanager
def silence_yolov5_logger():
    """Temporarily suppress YOLOv5's INFO logs during model construction."""
    logger = logging.getLogger(YOLOV5_LOGGER_NAME)
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)
