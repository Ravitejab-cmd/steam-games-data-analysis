"""
Steam Games Data Analysis
=========================
Exploratory analysis of 27,075 Steam store games (1997-2019).

Dataset: "Steam Store Games" by Nik Davis (Kaggle), CC-BY 4.0
          https://www.kaggle.com/datasets/nikdavis/steam-store-games

Run:
    pip install -r requirements.txt
    python analysis.py

Outputs: charts/*.png + a printed summary of key findings.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------- constants
DATA_PATH = Path(__file__).parent / "data" / "steam.csv"
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Publication-friendly style
sns.set_theme(style="whitegrid", context="talk")
PALETTE = "viridis"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

MIN_REVIEWS = 50  # review-count floor so tiny games don't skew sentiment stats


# ---------------------------------------------------------------- loading
def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load the CSV and derive the analysis columns."""
    df = pd.read_csv(path)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

    # Review sentiment
    df["total_reviews"] = df["positive_ratings"] + df["negative_ratings"]
    df["pos_ratio"] = np.where(
        df["total_reviews"] > 0,
        df["positive_ratings"] / df["total_reviews"],
        np.nan,
    )

    # Primary genre = first genre listed by Steam
    df["primary_genre"] = df["genres"].fillna("Unknown").str.split(";").str[0]

    df["is_free"] = df["price"] == 0
    return df


# ---------------------------------------------------------------- charts
def chart_releases_per_year(df: pd.DataFrame) -> None:
    """Games released per year (2005+, where the catalog has real volume)."""
    yearly = df[df["year"] >= 2005]["year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(yearly.index, yearly.values, color=sns.color_palette(PALETTE, len(yearly)))
    ax.set_title("Steam game releases per year (2005-2019)", fontsize=18, pad=14)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Games released")
    # Annotate the two biggest years
    for yr in yearly.nlargest(2).index:
        ax.annotate(f"{int(yearly[yr]):,}", (yr, yearly[yr]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=11, fontweight="bold")
    fig.savefig(CHARTS_DIR / "01_releases_per_year.png")
    plt.close(fig)


def chart_genre_distribution(df: pd.DataFrame) -> None:
    """Top 10 primary genres by number of games."""
    top = df["primary_genre"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette=PALETTE, hue=top.index,
                legend=False)
    ax.set_title("Top 10 genres by number of games", fontsize=18, pad=14)
    ax.set_xlabel("Number of games")
    ax.set_ylabel("")
    for i, v in enumerate(top.values):
        ax.text(v + 60, i, f"{v:,}", va="center", fontsize=11)
    fig.savefig(CHARTS_DIR / "02_genre_distribution.png")
    plt.close(fig)


def chart_price_vs_sentiment(df: pd.DataFrame) -> None:
    """Average positive-review ratio by price band (games with 50+ reviews)."""
    sub = df[df["total_reviews"] >= MIN_REVIEWS].copy()
    sub["band"] = "paid"
    sub.loc[sub["is_free"], "band"] = "Free"
    sub.loc[(sub["price"] > 0) & (sub["price"] < 5), "band"] = "Under $5"
    sub.loc[(sub["price"] >= 5) & (sub["price"] < 10), "band"] = "$5-10"
    sub.loc[(sub["price"] >= 10) & (sub["price"] < 20), "band"] = "$10-20"
    sub.loc[(sub["price"] >= 20) & (sub["price"] < 30), "band"] = "$20-30"
    sub.loc[sub["price"] >= 30, "band"] = "$30+"

    order = ["Free", "Under $5", "$5-10", "$10-20", "$20-30", "$30+"]
    agg = sub.groupby("band")["pos_ratio"].agg(["mean", "count"]).reindex(order)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(agg.index, agg["mean"], color=sns.color_palette(PALETTE, len(agg)))
    ax.set_ylim(0.60, 0.85)
    ax.set_title("Average positive-review ratio by price band\n(games with 50+ reviews)",
                 fontsize=18, pad=14)
    ax.set_xlabel("Price band")
    ax.set_ylabel("Avg. positive-review ratio")
    for bar, v, n in zip(bars, agg["mean"], agg["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{v:.3f}\n(n={n:,})", ha="center", fontsize=10)
    fig.savefig(CHARTS_DIR / "03_price_vs_sentiment.png")
    plt.close(fig)


def chart_genre_sentiment(df: pd.DataFrame) -> None:
    """Average positive-review ratio for the most-reviewed genres."""
    sub = df[df["total_reviews"] >= MIN_REVIEWS]
    by_genre = sub.groupby("primary_genre")["pos_ratio"].agg(["mean", "count"])
    top = by_genre.nlargest(8, "count").sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top["mean"], y=top.index, ax=ax, palette=PALETTE, hue=top.index,
                legend=False)
    ax.set_xlim(0.65, 0.85)
    ax.set_title("Average positive-review ratio by genre\n(8 most-reviewed genres, 50+ reviews)",
                 fontsize=18, pad=14)
    ax.set_xlabel("Avg. positive-review ratio")
    ax.set_ylabel("")
    for i, (g, row) in enumerate(top.iterrows()):
        ax.text(row["mean"] + 0.002, i, f"{row['mean']:.3f}", va="center", fontsize=11)
    fig.savefig(CHARTS_DIR / "04_genre_sentiment.png")
    plt.close(fig)


def chart_owners_vs_sentiment(df: pd.DataFrame) -> None:
    """Average positive-review ratio by estimated owners range."""
    order = ["0-20000", "20000-50000", "50000-100000", "100000-200000",
             "200000-500000", "500000-1000000", "1000000-2000000",
             "2000000-5000000", "5000000-10000000", "10000000-20000000"]
    sub = df[(df["total_reviews"] >= MIN_REVIEWS) & (df["owners"].isin(order))]
    agg = sub.groupby("owners")["pos_ratio"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(agg)), agg.values, color=sns.color_palette(PALETTE, len(agg)))
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([o.replace("-", "-\n") for o in agg.index], fontsize=10)
    ax.set_title("Average positive-review ratio by estimated owners range\n"
                 "(games with 50+ reviews)", fontsize=18, pad=14)
    ax.set_xlabel("Estimated owners range")
    ax.set_ylabel("Avg. positive-review ratio")
    ax.set_ylim(0.6, 0.9)
    fig.savefig(CHARTS_DIR / "05_owners_vs_sentiment.png")
    plt.close(fig)


# ---------------------------------------------------------------- findings
def print_findings(df: pd.DataFrame) -> None:
    """Print the headline numbers (mirrored in the README)."""
    sub = df[df["total_reviews"] >= MIN_REVIEWS]
    peak_year = df["year"].value_counts().idxmax()

    print("=" * 64)
    print("STEAM GAMES DATA ANALYSIS - KEY FINDINGS")
    print("=" * 64)
    print(f"Games analyzed            : {len(df):,}")
    print(f"Catalog period            : {df['release_date'].min().date()} to "
          f"{df['release_date'].max().date()}")
    print(f"Free-to-play share        : {df['is_free'].mean()*100:.1f}%")
    print(f"Median price (paid games) : ${df.loc[~df['is_free'], 'price'].median():.2f}")
    print(f"Peak release year         : {int(peak_year)} "
          f"({int(df['year'].value_counts().max()):,} games)")
    print(f"Top genre                 : {df['primary_genre'].value_counts().idxmax()} "
          f"({df['primary_genre'].value_counts().iloc[0]:,} games)")
    print(f"Overall avg pos. ratio    : {sub['pos_ratio'].mean():.3f} "
          f"(games with {MIN_REVIEWS}+ reviews)")
    print()
    print("Top 5 games by review volume:")
    top5 = df.nlargest(5, "total_reviews")[["name", "total_reviews", "pos_ratio", "price"]]
    for _, r in top5.iterrows():
        print(f"  - {r['name']}: {int(r['total_reviews']):,} reviews, "
              f"{r['pos_ratio']:.1%} positive, ${r['price']:.2f}")
    print()
    print("Price band vs avg positive ratio (50+ reviews):")
    sub2 = sub.copy()
    sub2["band"] = pd.cut(sub2["price"],
                          bins=[-0.01, 0.01, 5, 10, 20, 30, 1e9],
                          labels=["Free", "Under $5", "$5-10", "$10-20", "$20-30", "$30+"])
    print(sub2.groupby("band", observed=True)["pos_ratio"].mean().round(3).to_string())
    print()
    print(f"Price-rating correlation  : "
          f"{sub.loc[~sub['is_free'], ['price', 'pos_ratio']].corr().iloc[0,1]:.3f} "
          "(weak positive)")


def main() -> None:
    df = load_and_prepare(DATA_PATH)
    chart_releases_per_year(df)
    chart_genre_distribution(df)
    chart_price_vs_sentiment(df)
    chart_genre_sentiment(df)
    chart_owners_vs_sentiment(df)
    print_findings(df)
    print(f"\nCharts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
