"""Deterministic image transforms used by OCR evidence passes."""

from __future__ import annotations

import cv2
import numpy as np


def resize(image: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def sharpen(image: np.ndarray) -> np.ndarray:
    g = gray(image)
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.addWeighted(g, 1.65, blur, -0.65, 0)


def otsu(image: np.ndarray) -> np.ndarray:
    g = gray(image)
    _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def adaptive(image: np.ndarray) -> np.ndarray:
    g = gray(image)
    return cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        15,
    )
