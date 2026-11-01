"""Fingerprint a comic page image, so a re-uploaded chapter can be recognised.

The site re-lists ``144`` and ``144.0`` as separate chapters. Measured on the
real pages, the two pairs differ by **0 bits out of 256** on pages 2-10, so an
average hash with a small distance threshold identifies them reliably.

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
