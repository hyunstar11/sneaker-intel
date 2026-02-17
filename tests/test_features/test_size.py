"""Tests for size normalization behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from sneaker_intel.features.size import SizeNormalizer


def test_size_normalizer_reuses_training_distribution() -> None:
    normalizer = SizeNormalizer()

    train_df = pd.DataFrame({"Shoe Size": [8.0, 8.0, 10.0]})
    train_result = normalizer.extract(train_df)
    assert train_result["size_freq"].tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3])

    infer_df = pd.DataFrame({"Shoe Size": [10.0, 12.0]})
    infer_result = normalizer.extract(infer_df)
    assert infer_result["size_freq"].tolist() == pytest.approx([1 / 3, 0.0])


def test_size_normalizer_uses_unknown_frequency_when_unfitted() -> None:
    normalizer = SizeNormalizer(unknown_frequency=0.25)
    single_row = pd.DataFrame({"Shoe Size": [9.0]})
    result = normalizer.extract(single_row)
    assert result["size_freq"].tolist() == [0.25]
