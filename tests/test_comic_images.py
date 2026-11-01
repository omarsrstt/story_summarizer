# tests/test_comic_images.py
import io

import numpy as np
import pytest
from PIL import Image

from comic_images import (
    DUPLICATE_DISTANCE,
    average_hash,
    hamming,
    looks_like_image,
)


def _png(pixels) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_the_same_picture_hashes_identically():
    payload = _png(np.full((32, 32), 200))
    assert hamming(average_hash(payload), average_hash(payload)) == 0


def test_a_different_picture_hashes_further_than_the_threshold():
    white = average_hash(_png(np.full((32, 32), 255)))
    split = average_hash(_png(np.hstack([np.zeros((32, 16)), np.full((32, 16), 255)])))
    assert hamming(white, split) > DUPLICATE_DISTANCE


def test_a_rescale_of_the_same_picture_still_matches():
    plain = average_hash(_png(np.full((32, 32), 200)))
    bigger = average_hash(_png(np.full((128, 128), 200)))
    assert hamming(plain, bigger) <= DUPLICATE_DISTANCE


def test_an_html_error_page_is_not_an_image():
    assert not looks_like_image(b"<html>error</html>")


def test_each_real_format_is_recognised():
    for magic in (b"\xff\xd8\xff\xe0", b"\x89PNG\r\n\x1a\n", b"RIFF0000WEBP", b"GIF89a"):
        assert looks_like_image(magic + b"0" * 32)


def test_hashing_something_that_is_not_an_image_is_refused():
    with pytest.raises(ValueError):
        average_hash(b"<html>error</html>")
