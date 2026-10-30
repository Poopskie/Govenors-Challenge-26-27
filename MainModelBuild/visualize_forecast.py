"""Create a four-panel BVAR forecast chart from the downloaded data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model import BVARConfig, BayesianVAR


PLOTS = (
    ("ca_gdp_growth", "Canadian GDP growth", "Percent, annualized"),
    ("headline_inflation", "Headline inflation", "Percent, year over year"),
    ("core_inflation", "Core inflation proxy", "Percent, year over year"),
    ("overnight_rate", "Overnight rate", "Percent"),
)


def create_forecast_chart(
    data: pd.DataFrame,
    forecast: pd.DataFrame,
    output: Path,
) -> None:
    """Save historical values, forecast medians, and 80% bands to a PNG."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    data_index = (
        data.index.to_timestamp()
        if isinstance(data.index, pd.PeriodIndex)
        else data.index
    )
    forecast_index = (
        forecast.index.to_timestamp()
        if isinstance(forecast.index, pd.PeriodIndex)
        else forecast.index
    )

    for axis, (variable, title, ylabel) in zip(axes.flat, PLOTS):
        axis.plot(
            data_index,
            data[variable],
            color="#1f2937",
            linewidth=1.5,
            label="Historical",
        )
        axis.plot(
            forecast_index,
            forecast[f"{variable}_mean"],
            color="#2563eb",
            linewidth=2,
            label="BVAR median",
        )
        axis.fill_between(
            forecast_index,
            forecast[f"{variable}_lower80"],
            forecast[f"{variable}_upper80"],
            color="#60a5fa",
            alpha=0.25,
            label="80% interval",
        )
        axis.axvline(data_index[-1], color="#6b7280", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle("Bayesian VAR: Canadian growth and inflation outlook", fontsize=15)
    figure.text(
        0.01,
        0.01,
        "Dashed line marks the forecast origin. Core inflation is currently a total-CPI proxy.",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data") / "bvar_quarterly.csv")
    parser.add_argument("--output", type=Path, default=Path("output") / "bvar_forecast.png")
    parser.add_argument("--draws", type=int, default=2_000)
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    model = BayesianVAR(BVARConfig(draws=args.draws)).fit(data)
    forecast = model.forecast()
    create_forecast_chart(data, forecast, args.output)
    print(f"Saved forecast chart to {args.output}")


if __name__ == "__main__":
    main()
