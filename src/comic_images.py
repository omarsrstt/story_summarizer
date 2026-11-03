"""Fingerprint a comic chapter, so a re-uploaded chapter can be recognised.

The site re-lists ``144`` and ``144.0`` as separate chapters. Measured on the
real pages, those two are **0 bits apart out of 768** once pages 2-4 are
compared, while two distinct chapters that share an opener page (``175.1`` and
``175.2``) are **337 bits apart**.

Page 1 is not that signal, in either direction: the duplicate pair
``144``/``144.0`` was measured 88 bits apart on page 1, and the distinct pair
``175.1``/``175.2`` was 0 bits apart there because they share a cover. Hashing
page 1 therefore missed real duplicates and invented false ones -- which is what
``chapter_fingerprint`` fixes by starting at page 2.

This module is the reusable half of that: it knows nothing about scraping,
chapters or the manifest. It takes bytes and returns a string.

The hash is a 16x16 greyscale average hash -- 256 bits packed into 64 hex
characters. That exact width is part of the contract: the hashes are stored in
the manifest JSON, and a width that drifted would silently stop matching
everything already recorded.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

#: Magic bytes, so an error page served with a ``.jpg`` name is not kept.
#: Prefix tests, not full headers: every format here is told apart by its first
#: few bytes, and a real JPEG/PNG/WEBP/GIF all pass.
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8")

#: Bits two hashes may differ by and still be the same picture. The real
#: duplicate pairs measured 0, so this is slack for a re-encode, not for a
#: redraw -- a different picture lands far above it.
DUPLICATE_DISTANCE = 10

#: Side of the greyscale thumbnail the hash is taken from: 16x16 = 256 bits.
_HASH_SIZE = 16


def looks_like_image(payload: bytes) -> bool:
    """Whether ``payload`` starts with a known image format's magic bytes.

    Cheaper and more reliable than decoding: a Cloudflare or 404 page saved
    under ``.jpg`` is refused here without paying for a failed ``Image.open``.
    """

    return any(payload.startswith(magic) for magic in IMAGE_MAGIC)


def average_hash(payload: bytes) -> str:
    """``payload`` as a 16x16 average hash: 64 hex characters, 256 bits.

    Each pixel is compared with the image's mean brightness, and the resulting
    bitfield is packed into one integer. Comparing against the mean rather than
    an absolute level is what makes the hash survive a rescale or a re-encode
    while still separating genuinely different pages.

    Raises ``ValueError`` when the payload is not a decodable image. That is
    deliberate: an unreadable page must fail loudly here, where the message can
    name the payload, instead of being stored as a hash that matches nothing.
    """

    try:
        image = Image.open(io.BytesIO(payload)).convert("L")
        thumbnail = image.resize((_HASH_SIZE, _HASH_SIZE))
        pixels = np.asarray(thumbnail, dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            f"cannot read a page image from {len(payload)} bytes: {exc}"
        ) from exc

    bits = pixels > pixels.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:064x}"


def hamming(a: str, b: str) -> int:
    """How many bits differ between two hex hashes.

    Plain integers rather than per-character comparison: the hashes are packed
    bitfields, so an XOR and a popcount is the whole operation.
    """

    return bin(int(a, 16) ^ int(b, 16)).count("1")


def chapter_fingerprint(pages: list[bytes], *, skip: int = 1, take: int = 3) -> str:
    """One chapter's fingerprint: up to ``take`` page hashes concatenated.

    ``skip`` defaults to 1 because **page 1 is not evidence of a duplicate**,
    in either direction:

    * Two chapters that really are the same chapter were measured **88 bits
      apart on page 1** (``144``/``144.0``) -- the re-upload starts at a
      different opener, so hashing page 1 missed the duplicate entirely.
    * Two chapters that are genuinely different were measured **0 bits apart on
      page 1** (``175.1``/``175.2``) because they share a cover -- so hashing
      page 1 marked one as a duplicate of the other and dropped a real,
      four-page chapter.

    Pages 2 onwards separate them completely: 0/768 bits for the duplicate
    pairs, 337/768 for the distinct pair. That is the discriminating signal.

    Each hash is 64 hex characters, so the default ``take=3`` yields 192 hex
    characters = 768 bits. A caller comparing two fingerprints must compare
    whole fingerprints of the same length -- see ``comic_scraper`` for the
    threshold, which scales with that length.

    A page that will not decode is skipped rather than raised on: a response
    cut off mid-transfer is routine at this stage, and one bad page must not
    cost the whole fingerprint. ``""`` means no page was hashable at all, which
    callers treat as "not a candidate" rather than as a value two chapters can
    share.

    Note the asymmetry with ``skip`` and ``take``, which is deliberate: a page
    that fails to *fetch* arrives as ``b""`` and keeps its slot in ``pages``
    (the caller appends the placeholder), but a page that fails to *decode* is
    dropped here entirely. So a chapter whose page 2 will not decode yields the
    hashes of pages 3 and 4 -- 128 characters, not the 192 a clean scan gives --
    and a shorter fingerprint never compares with a longer one. That direction is
    the safe one: the cost is a missed duplicate (bandwidth), never a false one
    (a real chapter silently dropped). Keeping the slot instead would make the
    two scans comparable, at the risk of comparing a hole against a picture.
    """

    hashes: list[str] = []
    for payload in pages[skip:skip + take]:
        try:
            hashes.append(average_hash(payload))
        except ValueError:
            # Truncated or otherwise undecodable: skip this page, keep the rest.
            continue
    return "".join(hashes)
