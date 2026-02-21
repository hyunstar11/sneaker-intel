"""Launch demand tier segmentation using aftermarket signals."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class DemandSegmenter:
    """Segment sneakers into demand tiers using KMeans clustering."""

    def __init__(self, n_tiers: int = 3, random_state: int = 42):
        self.n_tiers = n_tiers
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._kmeans = KMeans(n_clusters=n_tiers, random_state=random_state, n_init=10)
        self._cluster_features = [
            "salesThisPeriod",
            "deadstockSold",
            "numberOfBids",
            "pricePremium",
        ]

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'demand_tier' column (High/Medium/Low) to the DataFrame."""
        df = df.copy()
        import numpy as np

        features = df[self._cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        scaled = self._scaler.fit_transform(features)
        labels = self._kmeans.fit_predict(scaled)

        # Map cluster IDs to tier names based on average salesThisPeriod
        cluster_means = (
            pd.DataFrame({"cluster": labels, "sales": df["salesThisPeriod"].fillna(0)})
            .groupby("cluster")["sales"]
            .mean()
            .sort_values()
        )
        tier_map = {}
        tier_names = ["Low", "Medium", "High"]
        for i, cluster_id in enumerate(cluster_means.index):
            tier_map[cluster_id] = tier_names[i] if i < len(tier_names) else f"Tier_{i}"

        df["demand_tier"] = pd.Series(labels).map(tier_map).values
        return df

    def get_tier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize each demand tier."""
        if "demand_tier" not in df.columns:
            df = self.fit_predict(df)

        summary_cols = [
            "salesThisPeriod",
            "deadstockSold",
            "numberOfBids",
            "pricePremium",
            "retail",
            "lowestAsk",
        ]
        existing = [c for c in summary_cols if c in df.columns]
        return df.groupby("demand_tier")[existing].agg(["mean", "median", "count"]).round(2)


def analyze_demand_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify which product characteristics predict high demand.

    Trains a simple classifier on the demand tier and returns feature importances.
    """
    from sklearn.ensemble import RandomForestClassifier

    if "demand_tier" not in df.columns:
        segmenter = DemandSegmenter()
        df = segmenter.fit_predict(df)

    feature_cols = [
        "retail",
        "volatility",
        "pricePremium",
        "numberOfAsks",
        "numberOfBids",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    import numpy as np

    X = df[existing].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["demand_tier"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    return (
        pd.DataFrame(
            {
                "Feature": existing,
                "Importance": clf.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
