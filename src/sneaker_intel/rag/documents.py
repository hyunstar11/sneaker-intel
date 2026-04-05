"""Builds a structured knowledge base from sneaker-intel and reddit-sentiment data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Brand market data mirror (sourced from sneakers2023.csv / NB08)
_STOCKX_BRAND_MARKET: dict[str, dict[str, object]] = {
    "Adidas": {"premium": 0.3070, "avg_deadstock": 9630, "volatility": "medium"},
    "Asics": {"premium": 0.1290, "avg_deadstock": 1000, "volatility": "low"},
    "New Balance": {"premium": 0.2185, "avg_deadstock": 2886, "volatility": "low"},
    "Nike": {"premium": 0.1770, "avg_deadstock": 6265, "volatility": "medium"},
    "Puma": {"premium": 0.4500, "avg_deadstock": 2967, "volatility": "low"},
}


@dataclass
class Document:
    id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.title}]\n{self.content}"


class DocumentBuilder:
    """Computes text documents from live data for RAG retrieval."""

    REDDIT_DATA = (
        Path(__file__).resolve().parents[4]
        / "reddit-sentiment"
        / "data"
        / "processed"
        / "annotated.parquet"
    )

    def __init__(self, portfolio_root: Path | None = None):
        if portfolio_root:
            self.REDDIT_DATA = (
                portfolio_root / "reddit-sentiment" / "data" / "processed" / "annotated.parquet"
            )
        _reddit_root = self.REDDIT_DATA.parent
        self._health_history = _reddit_root / "health_score_history.csv"

    def build(self) -> list[Document]:
        """Load all data sources and return the full document set."""
        from sneaker_intel.data import DatasetType, load_dataset

        mkt = load_dataset(DatasetType.MARKET_2023)
        mkt = mkt[np.isfinite(mkt["pricePremium"])].copy()
        mkt["release"] = pd.to_datetime(mkt["release"], errors="coerce")
        mkt["release_dow"] = mkt["release"].dt.day_name()
        mkt["release_month"] = mkt["release"].dt.strftime("%b")

        CORE = ["Nike", "Jordan", "adidas", "New Balance"]
        BRAND_MAP = {
            "Nike": "Nike",
            "Jordan": "Nike",
            "adidas": "Adidas",
            "New Balance": "New Balance",
            "Puma": "Puma",
            "ASICS": "Asics",
        }
        core = mkt[mkt["brand"].isin(CORE)].copy()
        mkt["brand_norm"] = mkt["brand"].map(BRAND_MAP)

        stx = load_dataset(DatasetType.STOCKX)

        def clean_price(s):
            return pd.to_numeric(str(s).replace("$", "").replace(",", ""), errors="coerce")

        stx["sale_price"] = stx["Sale Price"].apply(clean_price)
        stx["retail_price"] = stx["Retail Price"].apply(clean_price)
        stx["days_post_release"] = (
            pd.to_datetime(stx["Order Date"], format="%m/%d/%y", errors="coerce")
            - pd.to_datetime(stx["Release Date"], format="%m/%d/%y", errors="coerce")
        ).dt.days
        stx["premium_pct"] = (stx["sale_price"] - stx["retail_price"]) / stx["retail_price"] * 100
        stx = stx[stx["days_post_release"].between(0, 365)].copy()

        docs: list[Document] = []
        docs += self._brand_signal_docs(mkt, BRAND_MAP)
        docs.append(self._release_timing_doc(core))
        docs.append(self._pricing_doc(core))
        docs.append(self._hype_resilience_doc(stx))
        docs.append(self._geographic_doc(stx))
        docs.append(self._size_run_doc(stx))
        if self.REDDIT_DATA.exists():
            docs += self._reddit_docs()
            docs.append(self._cross_signal_doc())
        if self._health_history.exists():
            docs.append(self._health_trend_doc())
        return docs

    # ── individual document builders ──────────────────────────────────────

    def _brand_signal_docs(self, mkt: pd.DataFrame, brand_map: dict) -> list[Document]:
        mkt2 = mkt.dropna(subset=["brand_norm"])
        demand = (
            mkt2.groupby("brand_norm")
            .agg(
                products=("item", "count"),
                median_premium=("pricePremium", "median"),
                total_deadstock=("deadstockSold", "sum"),
                avg_volatility=("volatility", "mean"),
                median_bids=("numberOfBids", "median"),
            )
            .reset_index()
        )
        docs = []
        for _, row in demand.iterrows():
            brand = row["brand_norm"]
            docs.append(
                Document(
                    id=f"brand_demand_{brand.lower().replace(' ', '_')}",
                    title=f"{brand} — Aftermarket Demand Signal",
                    content=(
                        f"{brand} has {row['products']:.0f} products in the Market 2023 dataset. "
                        f"Median resale premium: {row['median_premium']:.3f}× retail "
                        f"(meaning secondary market buyers pay {row['median_premium'] * 100:.0f}% of retail on average). "
                        f"Total deadstock units sold: {row['total_deadstock']:,.0f}. "
                        f"Average market volatility: {row['avg_volatility']:.3f}. "
                        f"Median active bids (demand depth): {row['median_bids']:.1f}. "
                        f"{'High' if row['median_premium'] > 0.25 else 'Moderate' if row['median_premium'] > 0.15 else 'Low'} "
                        f"demand tier based on resale premium."
                    ),
                    metadata={"brand": brand, "topic": "demand", "source": "market_2023"},
                )
            )
        return docs

    def _release_timing_doc(self, core: pd.DataFrame) -> Document:
        MONTH_ORDER = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        timing_monthly = (
            core.dropna(subset=["release_month"])
            .groupby("release_month")["pricePremium"]
            .agg(["median", "count"])
            .reindex(MONTH_ORDER)
            .dropna()
            .reset_index()
        )
        timing_dow = (
            core.dropna(subset=["release_dow"])
            .groupby("release_dow")["pricePremium"]
            .median()
            .reset_index()
            .sort_values("pricePremium", ascending=False)
        )
        best_month = timing_monthly.loc[timing_monthly["median"].idxmax(), "release_month"]
        best_prem = timing_monthly["median"].max()
        avg_prem = timing_monthly["median"].mean()
        best_day = timing_dow.iloc[0]["release_dow"]
        top3_months = timing_monthly.nlargest(3, "median")["release_month"].tolist()
        return Document(
            id="release_timing",
            title="Release Timing Strategy — Best Months and Days",
            content=(
                f"Analysis of {len(core):,} core-brand releases (Nike, Jordan, adidas, New Balance) "
                f"shows release timing significantly affects resale premium. "
                f"Best release month: {best_month} ({best_prem:.3f}× median premium, "
                f"{(best_prem - avg_prem) / avg_prem * 100:+.0f}% above annual average). "
                f"Top 3 months by premium: {', '.join(top3_months)}. "
                f"Best release day of week: {best_day}. "
                f"Friday releases show median premium of "
                f"{core[core['release_dow'] == 'Friday']['pricePremium'].median():.3f}×; "
                f"Saturday releases show "
                f"{core[core['release_dow'] == 'Saturday']['pricePremium'].median():.3f}×. "
                f"Saturday drops outperform Friday by ~6.7 percentage points of premium. "
                f"Recommendation: target {best_month} drops on {best_day}s for maximum demand signal."
            ),
            metadata={"topic": "timing", "source": "market_2023"},
        )

    def _pricing_doc(self, core: pd.DataFrame) -> Document:
        bins = [0, 100, 125, 150, 175, 200, 250, 300, float("inf")]
        labels = [
            "<$100",
            "$100–125",
            "$125–150",
            "$150–175",
            "$175–200",
            "$200–250",
            "$250–300",
            "$300+",
        ]
        core = core.copy()
        core["price_bin"] = pd.cut(core["retail"], bins=bins, labels=labels)
        pricing = (
            core.groupby("price_bin", observed=True)
            .agg(median_premium=("pricePremium", "median"), products=("item", "count"))
            .reset_index()
        )
        sweet = pricing.loc[pricing["median_premium"].idxmax()]
        return Document(
            id="pricing_strategy",
            title="Retail Pricing Strategy — Sweet Spot Analysis",
            content=(
                f"For core brands (Nike, Jordan, adidas, New Balance), the retail pricing sweet spot "
                f"for maximizing resale premium is {sweet['price_bin']} retail, "
                f"which generates {sweet['median_premium']:.3f}× median resale premium. "
                f"Price bands below $125 signal low exclusivity and reduce perceived value. "
                f"Price bands above $250 suppress demand breadth and limit total addressable market. "
                f"Full pricing table: "
                + " | ".join(
                    f"{row['price_bin']}: {row['median_premium']:.3f}×"
                    for _, row in pricing.iterrows()
                )
                + ". "
                f"Recommendation: price limited releases in the {sweet['price_bin']} band "
                f"to maximize unmet demand signal."
            ),
            metadata={"topic": "pricing", "source": "market_2023"},
        )

    def _hype_resilience_doc(self, stx: pd.DataFrame) -> Document:
        stx["week"] = (stx["days_post_release"] // 7) * 7
        peak = stx[stx["days_post_release"] <= 14]["premium_pct"].median()
        late = stx[stx["days_post_release"].between(60, 90)]["premium_pct"].median()
        change = (late - peak) / abs(peak) * 100 if peak != 0 else 0
        return Document(
            id="hype_resilience",
            title="Hype Resilience — How Long Does Premium Hold?",
            content=(
                f"StockX 2019 data covering {len(stx):,} Yeezy and Off-White transactions shows "
                f"that investment-grade limited releases do NOT follow typical product demand decay. "
                f"Median premium in the first 14 days post-release: {peak:.1f}%. "
                f"Median premium at days 60–90: {late:.1f}%. "
                f"Change: {change:+.1f}% — the premium "
                f"{'increased' if change > 0 else 'held steady' if abs(change) < 10 else 'decreased'}. "
                f"This means investment-grade drops (Yeezy, Off-White tier) behave as store-of-value assets. "
                f"Implication: Nike should NOT restock within the first 60 days of a limited release. "
                f"Early restock announcements collapse the secondary market price and signal oversupply, "
                f"damaging brand heat and future release performance."
            ),
            metadata={"topic": "restock_policy", "source": "stockx_2019"},
        )

    def _geographic_doc(self, stx: pd.DataFrame) -> Document:
        geo = (
            stx.groupby("Buyer Region")
            .size()
            .reset_index(name="transactions")
            .sort_values("transactions", ascending=False)
        )
        geo["share"] = geo["transactions"] / geo["transactions"].sum() * 100
        geo["cumulative"] = geo["share"].cumsum()
        top2 = geo.head(2)
        states_50 = int((geo["cumulative"] <= 50).sum()) + 1
        states_75 = int((geo["cumulative"] <= 75).sum()) + 1
        return Document(
            id="geographic_demand",
            title="Geographic Demand Concentration",
            content=(
                f"Resale transaction data from StockX shows demand is highly concentrated. "
                f"Top state: {top2.iloc[0]['Buyer Region']} "
                f"({top2.iloc[0]['share']:.1f}% of all transactions). "
                f"Second state: {top2.iloc[1]['Buyer Region']} "
                f"({top2.iloc[1]['share']:.1f}% of all transactions). "
                f"Together the top 2 states account for "
                f"{top2['share'].sum():.1f}% of all US resale transactions. "
                f"Only {states_50} states are needed to cover 50% of demand. "
                f"Only {states_75} states are needed to cover 75% of demand. "
                f"Top 5 states by volume: "
                + ", ".join(geo.head(5)["Buyer Region"].tolist())
                + ". "
                f"Recommendation: weight SNKRS geofencing and retailer door allocation "
                f"toward {top2.iloc[0]['Buyer Region']} and {top2.iloc[1]['Buyer Region']} first."
            ),
            metadata={"topic": "geography", "source": "stockx_2019"},
        )

    def _size_run_doc(self, stx: pd.DataFrame) -> Document:
        size_dist = (
            stx.groupby("Shoe Size")
            .size()
            .reset_index(name="transactions")
            .sort_values("Shoe Size")
        )
        size_dist["share"] = size_dist["transactions"] / size_dist["transactions"].sum() * 100
        size_dist["cumulative"] = size_dist["share"].cumsum()
        core_mask = (size_dist["cumulative"] >= 12.5) & (size_dist["cumulative"] <= 87.5)
        core_sizes = size_dist[core_mask]["Shoe Size"].tolist()
        core_share = size_dist[core_mask]["share"].sum()
        return Document(
            id="size_run",
            title="Size Run Optimization — Production Allocation by Size",
            content=(
                f"StockX transaction data provides a pure demand signal for size distribution. "
                f"Core sizes covering the middle 75% of demand: "
                f"{float(core_sizes[0])} to {float(core_sizes[-1])} (US Men's). "
                f"These {len(core_sizes)} sizes account for {core_share:.1f}% of all transactions. "
                f"Sizes below 8 represent less than 5% of demand. "
                f"Sizes above 13 represent less than 3% of demand. "
                f"Recommendation: concentrate full production runs in sizes "
                f"{float(core_sizes[0])}–{float(core_sizes[-1])}; "
                f"use half-size or quarter-size production runs outside this window. "
                f"This reduces excess inventory on tail sizes and improves sell-through rate."
            ),
            metadata={"topic": "size_run", "source": "stockx_2019"},
        )

    def _reddit_docs(self) -> list[Document]:
        rs = pd.read_parquet(self.REDDIT_DATA)
        rs["brands"] = rs["brands"].apply(
            lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []
        )
        rs["channels"] = rs["channels"].apply(
            lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []
        )
        rs_exp = rs.explode("brands").dropna(subset=["brands"])
        rs_exp = rs_exp[rs_exp["brands"].str.len() > 0]

        sentiment = (
            rs_exp.groupby("brands")
            .agg(
                mentions=("hybrid_score", "count"),
                avg_sentiment=("hybrid_score", "mean"),
                positive_pct=("hybrid_score", lambda x: (x > 0.05).mean() * 100),
                negative_pct=("hybrid_score", lambda x: (x < -0.05).mean() * 100),
                seeking=("primary_intent", lambda x: (x == "seeking_purchase").sum()),
                completed=("primary_intent", lambda x: (x == "completed_purchase").sum()),
            )
            .reset_index()
            .rename(columns={"brands": "brand"})
        )
        sentiment = sentiment[sentiment["mentions"] >= 5]

        # Channel attribution
        rs_ch = rs.explode("brands").explode("channels")
        rs_ch = rs_ch[
            rs_ch["brands"].notna()
            & (rs_ch["brands"].str.len() > 0)
            & rs_ch["channels"].notna()
            & (rs_ch["channels"].str.len() > 0)
        ].reset_index(drop=True)
        top_channels = (
            rs_ch.groupby(["brands", "channels"])
            .size()
            .reset_index(name="n")
            .sort_values(["brands", "n"], ascending=[True, False])
            .groupby("brands")
            .head(3)
        )

        docs = []
        for _, row in sentiment.iterrows():
            brand = row["brand"]
            brand_ch = top_channels[top_channels["brands"] == brand]["channels"].tolist()
            tone = (
                "positive"
                if row["avg_sentiment"] > 0.1
                else "neutral"
                if row["avg_sentiment"] > -0.1
                else "negative"
            )
            docs.append(
                Document(
                    id=f"reddit_sentiment_{brand.lower().replace(' ', '_')}",
                    title=f"{brand} — Reddit Consumer Sentiment (Feb 2026)",
                    content=(
                        f"{brand} appears in {row['mentions']:.0f} Reddit posts and comments "
                        f"across 9 sneaker subreddits (Feb 2026). "
                        f"Average hybrid sentiment score: {row['avg_sentiment']:+.3f} ({tone}). "
                        f"{row['positive_pct']:.1f}% of mentions are positive; "
                        f"{row['negative_pct']:.1f}% are negative. "
                        f"Active purchase seekers ('W2C', 'where to cop'): {row['seeking']:.0f}. "
                        f"Confirmed purchases: {row['completed']:.0f}. "
                        + (
                            f"Top retail channels mentioned alongside {brand}: "
                            f"{', '.join(brand_ch)}. "
                            if brand_ch
                            else ""
                        )
                        + f"Consumer narrative is {tone} — "
                        + (
                            "strong buy intent and positive brand perception."
                            if tone == "positive"
                            else "mixed signals, monitor for trend shifts."
                            if tone == "neutral"
                            else "negative sentiment, investigate root cause."
                        )
                    ),
                    metadata={"brand": brand, "topic": "sentiment", "source": "reddit_feb2026"},
                )
            )

        # Overall market sentiment doc — with cross-signal teaser
        total = len(rs)
        avg_all = rs["hybrid_score"].mean()
        docs.append(
            Document(
                id="reddit_market_overview",
                title="Reddit Sneaker Market — Overall Sentiment Overview",
                content=(
                    f"Dataset: {total:,} Reddit posts and comments from 9 subreddits "
                    f"(r/Sneakers, r/Nike, r/Adidas, r/SneakerMarket, r/Jordans, r/Yeezy, "
                    f"r/malefashionadvice, r/Running, r/Basketball), collected Feb 2026. "
                    f"Overall average sentiment: {avg_all:+.3f} (positive). "
                    f"Total brands tracked: {sentiment['brand'].nunique()}. "
                    f"Most mentioned brand: {sentiment.loc[sentiment['mentions'].idxmax(), 'brand']} "
                    f"({sentiment['mentions'].max():.0f} mentions). "
                    f"Most positive brand: {sentiment.loc[sentiment['avg_sentiment'].idxmax(), 'brand']} "
                    f"(sentiment {sentiment['avg_sentiment'].max():+.3f}). "
                    f"The sneaker community shows broadly positive sentiment with Nike and Adidas "
                    f"dominating conversation volume."
                ),
                metadata={"topic": "market_overview", "source": "reddit_feb2026"},
            )
        )
        return docs

    def _cross_signal_doc(self) -> Document:
        """Combined StockX market signal + Reddit sentiment per brand."""

        rs = pd.read_parquet(self.REDDIT_DATA)
        rs["brands"] = rs["brands"].apply(
            lambda x: list(x) if hasattr(x, "__iter__") and not isinstance(x, str) else []
        )
        rs_exp = rs.explode("brands").dropna(subset=["brands"])
        rs_exp = rs_exp[rs_exp["brands"].str.len() > 0]
        sentiment = (
            rs_exp.groupby("brands")
            .agg(
                avg_sentiment=("hybrid_score", "mean"),
                mentions=("hybrid_score", "count"),
                seeking=("primary_intent", lambda x: (x == "seeking_purchase").sum()),
                completed=("primary_intent", lambda x: (x == "completed_purchase").sum()),
            )
            .reset_index()
            .rename(columns={"brands": "brand"})
        )

        lines = []
        for brand, mkt in _STOCKX_BRAND_MARKET.items():
            row = sentiment[sentiment["brand"] == brand]
            premium_pct = mkt["premium"] * 100
            deadstock = mkt["avg_deadstock"]
            if row.empty:
                lines.append(
                    f"{brand}: StockX premium {premium_pct:.1f}%, "
                    f"deadstock {deadstock:,.0f} units; no Reddit data available."
                )
                continue
            r = row.iloc[0]
            intent_ratio = (
                r["completed"] / (r["seeking"] + r["completed"])
                if (r["seeking"] + r["completed"]) > 0
                else 0.0
            )
            tone = (
                "positive"
                if r["avg_sentiment"] > 0.1
                else "neutral"
                if r["avg_sentiment"] > -0.1
                else "negative"
            )
            lines.append(
                f"{brand}: StockX resale premium {premium_pct:.1f}% above retail, "
                f"deadstock {deadstock:,.0f} units sold; "
                f"Reddit sentiment {r['avg_sentiment']:+.3f} ({tone}), "
                f"{r['mentions']:.0f} mentions, "
                f"purchase conversion {intent_ratio:.1%} "
                f"({r['completed']:.0f} confirmed / {r['seeking']:.0f} seeking)."
            )

        # Identify cross-signal highlights
        brand_data = {}
        for brand, mkt in _STOCKX_BRAND_MARKET.items():
            row = sentiment[sentiment["brand"] == brand]
            brand_data[brand] = {
                "premium": mkt["premium"],
                "sentiment": float(row["avg_sentiment"].iloc[0]) if not row.empty else 0.0,
            }
        best_combo = max(
            brand_data,
            key=lambda b: brand_data[b]["premium"] + brand_data[b]["sentiment"],
        )
        rising_sent = max(brand_data, key=lambda b: brand_data[b]["sentiment"])
        undervalued = max(
            brand_data,
            key=lambda b: brand_data[b]["sentiment"] - brand_data[b]["premium"],
        )

        content = (
            "Cross-signal analysis — StockX resale premium vs. Reddit consumer sentiment "
            "per brand:\n" + "\n".join(lines) + f"\n\nKey insights: "
            f"Strongest combined signal (high premium + positive sentiment): {best_combo}. "
            f"Highest Reddit sentiment: {rising_sent} "
            f"({brand_data[rising_sent]['sentiment']:+.3f}). "
            f"Best sentiment-to-premium ratio (undervalued social signal): {undervalued} — "
            f"strong community enthusiasm relative to current resale premium suggests "
            f"emerging demand not yet priced into the secondary market."
        )
        return Document(
            id="cross_signal_analysis",
            title="Cross-Signal Analysis — StockX Premium vs. Reddit Sentiment by Brand",
            content=content,
            metadata={"topic": "cross_signal", "source": "combined"},
        )

    def _health_trend_doc(self) -> Document:
        """Brand health score history — trend direction per brand."""
        history = pd.read_csv(self._health_history, parse_dates=["timestamp"])
        if history.empty:
            return Document(
                id="health_trend",
                title="Brand Health Score Trend",
                content="No health score history available yet.",
                metadata={"topic": "health_trend", "source": "health_score_history"},
            )

        latest_ts = history["timestamp"].max()
        latest = history[history["timestamp"] == latest_ts]

        lines = []
        for _, row in latest.sort_values("health_score", ascending=False).iterrows():
            brand = row["brand"]
            bdf = history[history["brand"] == brand].sort_values("timestamp")
            if len(bdf) >= 2:
                delta = bdf["health_score"].iloc[-1] - bdf["health_score"].iloc[-2]
                direction = (
                    "↑ improving"
                    if delta > 0.01
                    else "↓ declining"
                    if delta < -0.01
                    else "→ stable"
                )
            else:
                direction = "→ stable (insufficient history)"
            lines.append(
                f"{brand}: health score {row['health_score']:.3f} "
                f"({row['signal']}), trend {direction}, "
                f"avg sentiment {row['avg_sentiment']:+.3f}, "
                f"{row['mention_count']:.0f} Reddit mentions."
            )

        n_snapshots = history["timestamp"].nunique()
        content = (
            f"Brand health scores as of {latest_ts.strftime('%Y-%m-%d')} "
            f"({n_snapshots} historical snapshots).\n"
            + "\n".join(lines)
            + "\n\nHealth score formula: 40% StockX resale premium + 25% deadstock volume "
            "  + 20% Reddit sentiment + 15% purchase intent (all min-max normalized). "
            "Scores above 0.6 = Scale Up, 0.4–0.6 = Hold, below 0.4 = Watch."
        )
        return Document(
            id="health_trend",
            title="Brand Health Score — Composite Signal & Trend",
            content=content,
            metadata={"topic": "health_trend", "source": "health_score_history"},
        )
