import json

import numpy as np

from benchmarks.legacy import legacy_arrays, load_legacy


def write(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "source": "test",
                "actions": ["tesseract", "easyocr", "merge"],
                "rows": {
                    "a.png": {
                        "split": "test",
                        "cer": [0.5, 0.1, 0.2],
                        "action": 1,
                        "time": [0.2, 0.05],
                    },
                    "b.png": {
                        "split": "test",
                        "cer": [0.0, 0.9, 0.3],
                        "action": 2,
                        "time": [0.1, 0.04],
                    },
                },
            }
        )
    )
    return path


def test_arrays_follow_the_requested_order(tmp_path):
    arrays = legacy_arrays(load_legacy(write(tmp_path)), ["b.png", "a.png"])
    assert arrays["present"].tolist() == [True, True]
    assert arrays["cer"].tolist() == [[0.0, 0.9, 0.3], [0.5, 0.1, 0.2]]
    assert arrays["action"].tolist() == [2, 1]
    # merge runs both engines; EasyOCR alone costs only its own time
    assert np.allclose(arrays["seconds"], [0.14, 0.05])


def test_unknown_images_are_marked_absent(tmp_path):
    arrays = legacy_arrays(load_legacy(write(tmp_path)), ["a.png", "new.png"])
    assert arrays["present"].tolist() == [True, False]
    assert np.isnan(arrays["cer"][1]).all()
    assert arrays["action"][1] == -1


def test_the_tracked_file_is_the_published_router():
    legacy = load_legacy()
    test = [row for row in legacy["rows"].values() if row["split"] == "test"]
    assert len(legacy["rows"]) == 3380 and len(test) == 600
    mean = np.mean([min(row["cer"][row["action"]], 1.0) for row in test])
    assert round(float(mean), 3) == 0.076
