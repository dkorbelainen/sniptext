"""The selector SnipText shipped before the router (commit bf3fde3), kept as a baseline.

It decided between "fast" (Tesseract) and "ensemble" (merge) from seven image
statistics: fixed thresholds first, then a classifier fitted on hand-made random
feature ranges for the cases in between.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageStat

_TESSERACT, _MERGE = 0, 2


def legacy_features(image: Image.Image) -> np.ndarray:
    """ImageAnalyzer.extract_features as of bf3fde3, including its uint8 arithmetic."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    stat = ImageStat.Stat(image)
    brightness = np.mean(stat.mean)
    contrast = np.mean(stat.stddev)
    gray = image.convert("L")
    pixels = np.array(gray)
    sharpness = (
        np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean()
        if pixels.shape[0] > 1 and pixels.shape[1] > 1
        else 0.0
    )
    text_density = float((pixels < 128).mean())
    smoothed = np.array(gray.filter(ImageFilter.SMOOTH), dtype=np.float32)
    noise_level = min(float(np.std(pixels.astype(np.float32) - smoothed)) / 30.0, 1.0)
    red, green, blue = stat.mean
    has_color = 1 if np.std([red, green, blue]) > 10 else 0
    width, height = image.size
    size_ratio = width / height if height > 0 else 1.0
    return np.array(
        [
            brightness / 255.0,
            min(contrast / 60.0, 1.0),
            min(sharpness / 30.0, 1.0),
            has_color,
            min(size_ratio, 5.0) / 5.0,
            text_density,
            noise_level,
        ]
    )


def _default_model(seed: int):
    """The bootstrap classifier: 60 'easy' and 40 'hard' rows drawn from fixed ranges."""
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.RandomState(seed)
    u = rng.uniform
    rows, labels = [], []
    for _ in range(60):
        rows.append([u(0.5, 0.85), u(0.4, 0.9), u(0.5, 1.0), rng.randint(0, 2), u(0.2, 0.9),
                     u(0.05, 0.30), u(0.0, 0.15)])  # fmt: skip
        labels.append(0)
    for _ in range(40):
        scenario = rng.choice(["low_contrast", "extreme_brightness", "blurry", "noisy"])
        if scenario == "low_contrast":
            row = [u(0.3, 0.7), u(0.05, 0.25), u(0.2, 0.6), rng.randint(0, 2), u(0.2, 0.9),
                   u(0.01, 0.4), u(0.1, 0.5)]  # fmt: skip
        elif scenario == "extreme_brightness":
            row = [rng.choice([u(0.05, 0.25), u(0.85, 1.0)]), u(0.15, 0.4), u(0.3, 0.7),
                   rng.randint(0, 2), u(0.2, 0.9), u(0.01, 0.5), u(0.05, 0.4)]  # fmt: skip
        elif scenario == "blurry":
            row = [u(0.3, 0.8), u(0.2, 0.5), u(0.05, 0.3), rng.randint(0, 2), u(0.2, 0.9),
                   u(0.01, 0.3), u(0.05, 0.35)]  # fmt: skip
        else:
            row = [u(0.3, 0.8), u(0.15, 0.45), u(0.2, 0.6), rng.randint(0, 2), u(0.2, 0.9),
                   u(0.01, 0.15), u(0.35, 0.9)]  # fmt: skip
        rows.append(row)
        labels.append(1)
    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
    )
    return model.fit(np.array(rows), np.array(labels))


def legacy_actions(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """Router action index (tesseract or merge) the old selector picks for each row."""
    features = np.atleast_2d(np.asarray(features, dtype=float))
    model = _default_model(seed)
    actions = np.empty(len(features), dtype=int)
    for i, row in enumerate(features):
        contrast, sharpness, density, noise = row[1], row[2], row[5], row[6]
        if contrast < 0.2 or sharpness < 0.2 or noise > 0.6 or density < 0.02:
            actions[i] = _MERGE
        elif contrast > 0.5 and sharpness > 0.4 and noise < 0.4:
            actions[i] = _TESSERACT
        else:
            actions[i] = _MERGE if model.predict(row.reshape(1, -1))[0] == 1 else _TESSERACT
    return actions
