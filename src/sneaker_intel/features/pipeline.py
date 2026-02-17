"""Feature pipeline that chains extractors together."""

from __future__ import annotations

import pandas as pd

from sneaker_intel.features.base import BaseFeatureExtractor


class FeaturePipeline:
    """Chain multiple feature extractors and apply them sequentially."""

    def __init__(self, extractors: list[BaseFeatureExtractor]):
        self._extractors = extractors

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all extractors in order."""
        for extractor in self._extractors:
            df = extractor.extract(df)
        return df

    @property
    def feature_names(self) -> list[str]:
        """Aggregate feature names from all extractors."""
        names: list[str] = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        return names

    @classmethod
    def stockx_default(cls) -> FeaturePipeline:
        """Create the default pipeline for the StockX dataset."""
        from sneaker_intel.features.color import ColorFeatureExtractor
        from sneaker_intel.features.region import RegionFeatureExtractor
        from sneaker_intel.features.size import SizeNormalizer
        from sneaker_intel.features.sneaker_type import SneakerTypeExtractor
        from sneaker_intel.features.temporal import TemporalFeatureExtractor

        return cls(
            [
                ColorFeatureExtractor(),
                SneakerTypeExtractor(),
                RegionFeatureExtractor(),
                SizeNormalizer(),
                TemporalFeatureExtractor(),
            ]
        )
