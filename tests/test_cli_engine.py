import sys

from scraper import parse_arguments

WEBSITES = [{"name": "Light Novel Pub"}]


def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["scraper.py", *argv])
    return parse_arguments(WEBSITES)


def test_engine_defaults_to_fast(monkeypatch):
    args = _parse(monkeypatch)
    assert args.engine == "fast"
    assert args.concurrency == 3
    assert args.overwrite is False


def test_engine_legacy_selectable(monkeypatch):
    assert _parse(monkeypatch, "--engine", "legacy").engine == "legacy"


def test_concurrency_and_overwrite(monkeypatch):
    args = _parse(monkeypatch, "--concurrency", "5", "--overwrite")
    assert args.concurrency == 5 and args.overwrite is True
