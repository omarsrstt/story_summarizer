# tests/test_comic_images.py
import io

import numpy as np
import pytest
from PIL import Image

from comic_images import (
    DUPLICATE_DISTANCE,
    average_hash,
    chapter_fingerprint,
    hamming,
    looks_like_image,
)


def _png(pixels) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


#: Three decodable pages, far apart on the average hash, built as plain pictures
#: rather than bit patterns so the hash really is computed from pixels as it is
#: in production. ``_quad`` gives a page a distinctive 2x2 brightness block so
#: two of them cannot collapse onto the same hash the way flat fills do.
def _quad(top_left, top_right, bottom_left, bottom_right) -> bytes:
    top = np.hstack([np.full((16, 16), top_left), np.full((16, 16), top_right)])
    bottom = np.hstack(
        [np.full((16, 16), bottom_left), np.full((16, 16), bottom_right)]
    )
    return _png(np.vstack([top, bottom]))


def _rescaled(payload: bytes, size: int) -> bytes:
    """The same picture at a different resolution, as a re-upload would be."""
    buffer = io.BytesIO()
    with Image.open(io.BytesIO(payload)) as image:
        image.resize((size, size), Image.NEAREST).save(buffer, format="PNG")
    return buffer.getvalue()


PAGE_A = _quad(255, 0, 0, 255)
PAGE_B = _quad(0, 255, 255, 0)
PAGE_C = _quad(255, 255, 0, 0)


def test_the_same_picture_hashes_identically():
    # Deliberately the same *file bytes* re-hashed, not a value compared with
    # itself: this pins that ``average_hash`` is deterministic, so a rescan of
    # an unchanged page cannot make a duplicate look like a new chapter.
    payload = _png(np.full((32, 32), 200))
    first = average_hash(payload)
    second = average_hash(payload)
    assert first == second
    assert hamming(first, second) == 0


def test_a_different_picture_hashes_further_than_the_threshold():
    white = average_hash(_png(np.full((32, 32), 255)))
    split = average_hash(_png(np.hstack([np.zeros((32, 16)), np.full((32, 16), 255)])))
    assert hamming(white, split) > DUPLICATE_DISTANCE


def test_a_rescale_of_the_same_picture_still_matches():
    # Structured quadrants, not flat fills: a flat fill sits below its own mean
    # and hashes to all-zeros at every size, so it would "match" under any
    # implementation and prove nothing about the resize path.
    plain = average_hash(_quad(255, 0, 0, 255))
    bigger = average_hash(_rescaled(_quad(255, 0, 0, 255), 128))
    assert hamming(plain, bigger) <= DUPLICATE_DISTANCE

    other = average_hash(_quad(0, 255, 255, 0))
    assert hamming(plain, other) > DUPLICATE_DISTANCE


def test_an_html_error_page_is_not_an_image():
    assert not looks_like_image(b"<html>error</html>")


def test_each_real_format_is_recognised():
    for magic in (b"\xff\xd8\xff\xe0", b"\x89PNG\r\n\x1a\n", b"RIFF0000WEBP", b"GIF89a"):
        assert looks_like_image(magic + b"0" * 32)


def test_hashing_something_that_is_not_an_image_is_refused():
    with pytest.raises(ValueError):
        average_hash(b"<html>error</html>")


# -- a chapter fingerprint is pages 2 onwards ---------------------------------


def test_a_fingerprint_is_three_page_hashes_concatenated():
    fingerprint = chapter_fingerprint([PAGE_A, PAGE_B, PAGE_C, PAGE_A])

    assert len(fingerprint) == 192
    # Pages 2, 3 and 4 -- page 1 is skipped, and the fourth page is past the
    # default ``take`` of 3.
    assert fingerprint == average_hash(PAGE_B) + average_hash(PAGE_C) + average_hash(PAGE_A)


def test_a_fingerprint_skips_page_one_so_a_shared_opener_cannot_match():
    # The live bug: 175.1 and 175.2 share a byte-identical opener page and are
    # otherwise different chapters. Hashing page 1 made them identical, because
    # both page-1 hashes are the same 64 characters, so the pair read as
    # distance 0.
    first = [PAGE_A, PAGE_B, PAGE_C, PAGE_A]
    second = [PAGE_A, PAGE_C, PAGE_A, PAGE_B]

    one = chapter_fingerprint(first)
    two = chapter_fingerprint(second)

    assert len(one) == len(two) == 192
    assert hamming(one, two) > 0

    # And the opener really is shared: hashing just page 1 gives the two
    # chapters the same value, which is what the old fingerprint compared.
    opener_of_first = chapter_fingerprint(first[:1], skip=0, take=1)
    opener_of_second = chapter_fingerprint(second[:1], skip=0, take=1)
    assert opener_of_first == opener_of_second == average_hash(PAGE_A)


def test_a_fingerprint_of_a_chapter_with_fewer_pages_is_shorter():
    assert len(chapter_fingerprint([PAGE_A])) == 0
    assert len(chapter_fingerprint([PAGE_A, PAGE_B])) == 64


def test_a_fingerprint_falls_back_to_nothing_when_no_page_decodes():
    assert chapter_fingerprint([PAGE_A, b"<html>error</html>", b""]) == ""


def test_one_undecodable_page_is_skipped_not_fatal():
    # A truncated response keeps its slot in the list -- this is what
    # ``scan_chapter`` does -- so the pages after it are still hashed.
    fingerprint = chapter_fingerprint([PAGE_A, b"<html>error</html>", PAGE_C, PAGE_A])

    assert fingerprint == average_hash(PAGE_C) + average_hash(PAGE_A)


def test_a_fingerprint_of_no_pages_is_empty():
    assert chapter_fingerprint([]) == ""


def test_skip_and_take_are_adjustable():
    assert chapter_fingerprint([PAGE_A, PAGE_B, PAGE_C], skip=0, take=1) == average_hash(PAGE_A)

