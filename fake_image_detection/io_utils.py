from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path, mode: str = "RGB") -> Optional[np.ndarray]:
    path = str(path)
    try:
        with Image.open(path) as im:
            if mode == "L":
                im = im.convert("L")
            elif mode == "RGB":
                im = im.convert("RGB")
            arr = np.array(im)
        if arr is None or arr.size == 0:
            return None
        return arr
    except Exception:
        pass
    try:
        flag = cv2.IMREAD_COLOR if mode == "RGB" else cv2.IMREAD_GRAYSCALE
        arr = cv2.imread(path, flag)
        if arr is None:
            return None
        if mode == "RGB":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    except Exception:
        return None


def is_corrupt(path) -> bool:
    return load_image(path) is None


def image_size(path) -> Optional[tuple]:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None
