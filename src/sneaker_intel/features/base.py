"""Base class for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors.

    Subclasses must implement `extract` and `feature_names`.
    """

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add new feature columns to the DataFrame.

        Args:
            df: Input DataFrame (modified in-place or copied).

        Returns:
            DataFrame with new feature columns added.
        """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature column names produced by this extractor."""
