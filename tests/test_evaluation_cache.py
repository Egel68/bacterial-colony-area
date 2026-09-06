from pathlib import Path

import pytest

from testing.cache import compute_dataset_signature


def test_signature_stable():
    sig1 = compute_dataset_signature("test_images")
    sig2 = compute_dataset_signature("test_images")
    assert sig1 == sig2
    assert len(sig1) == 64


def test_signature_changes():
    sig1 = compute_dataset_signature("test_images")
    sig2 = compute_dataset_signature("datasets/22022540_imported")
    assert sig1 != sig2


def test_cache_roundtrip(tmp_path):
    from testing.cache import load_cache, save_cache, compute_dataset_signature

    test_data = {
        "dataset_root": str(tmp_path),
        "dataset_signature": "abc123",
        "created": "2026-01-01T00:00:00",
        "algorithms": {
            "ClassicDefault": {
                "source": {"mean_iou": 0.72, "num_samples": 369},
                "cropped": {"mean_iou": 0.75, "num_samples": 369},
            }
        },
    }
    save_cache(str(tmp_path), test_data)
    loaded = load_cache(str(tmp_path))
    assert loaded is not None
    assert loaded["algorithms"]["ClassicDefault"]["source"]["mean_iou"] == 0.72


def test_cache_nonexistent(tmp_path):
    from testing.cache import load_cache

    loaded = load_cache(str(tmp_path))
    assert loaded is None