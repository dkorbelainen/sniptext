import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from benchmarks import realweb


def page(box, frame=3):
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 199, 9), fill="navy")
    draw.rectangle(box, outline=(255, 0, 0), width=frame)
    return image


def reds(image):
    r, g, b = np.asarray(image).astype(int).transpose(2, 0, 1)
    return int(((r > 180) & (g < 90) & (b < 90)).sum())


def test_strip_frame_cuts_a_drawn_frame():
    framed = page((0, 0, 199, 99))
    assert realweb.strip_frame(framed).size == (194, 94)
    assert reds(realweb.strip_frame(framed)) == 0


def test_strip_frame_cuts_a_frame_that_starts_inside_the_crop():
    shifted = page((1, 1, 198, 98))
    assert reds(realweb.strip_frame(shifted)) == 0
    assert realweb.strip_frame(shifted).size == (192, 92)


def test_strip_frame_keeps_an_image_without_a_frame():
    plain = Image.new("RGB", (80, 30), "white")
    assert realweb.strip_frame(plain).size == (80, 30)


def test_strip_frame_keeps_red_content_past_the_frame_width():
    assert realweb.strip_frame(Image.new("RGB", (80, 30), (255, 0, 0))).size == (68, 18)


def test_crop_takes_the_annotated_box_without_its_frame():
    image = page((40, 20, 139, 69))
    crop = realweb.crop(image, [0.2, 0.2, 0.7, 0.7])
    assert crop.size == (94, 44) and reds(crop) == 0


def test_generate_writes_one_crop_per_row(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buffer = io.BytesIO()
    page((40, 20, 139, 69)).save(buffer, format="PNG")
    rows = [
        {"id": f"element_ocr_{k}", "website": site, "image": {"bytes": buffer.getvalue()},
         "bbox": [0.2, 0.2, 0.7, 0.7], "answer": text}
        for k, (site, text) in enumerate((("a.org", "first text"), ("a.org", "second text")))
    ]  # fmt: skip
    source = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    samples = realweb.generate(tmp_path / "out", source)
    assert [s.gt for s in samples] == ["first text", "second text"]
    assert {(s.source, s.split, s.text_id, s.lang) for s in samples} == {
        ("visualwebbench", "real_web", "a.org", "en")
    }
    assert [s.path.name for s in samples] == ["element_ocr_0.png", "element_ocr_1.png"]
    with Image.open(samples[0].path) as saved:
        assert saved.size == (94, 44)


def test_a_changed_download_is_rejected(tmp_path):
    target = tmp_path / "rows.parquet"
    with pytest.raises(ValueError, match="checksum"):
        realweb.fetch(target, opener=lambda url: io.BytesIO(b"other bytes"))
    assert not target.exists()
