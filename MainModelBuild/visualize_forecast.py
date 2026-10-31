"""Create baseline and policy-scenario charts from the macro models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd

from gdpmodel import load_component_csv
from model import BVARConfig, BayesianVAR


PLOTS = (
    ("ca_gdp_growth", "Canadian GDP growth", "Percent, annualized"),
    ("headline_inflation", "Headline inflation", "Percent, year over year"),
    ("core_inflation", "Core inflation proxy", "Percent, year over year"),
)


def create_forecast_chart(
    data: pd.DataFrame,
    forecast: pd.DataFrame,
    output: Path,
    history_quarters: int | None = None,
) -> None:
    """Save historical values, forecast medians, and 80% bands to a PNG."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    if history_quarters is not None:
        if history_quarters < 1:
            raise ValueError("history_quarters must be positive")
        data = data.iloc[-history_quarters:]
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
    connected_forecast_index = forecast_index.insert(0, data_index[-1])

    for axis, (variable, title, ylabel) in zip(axes.flat, PLOTS):
        axis.plot(
            data_index,
            data[variable],
            color="#1f2937",
            linewidth=1.5,
            label="Historical",
        )
        axis.plot(
            connected_forecast_index,
            [data[variable].iloc[-1], *forecast[f"{variable}_mean"]],
            color="#2563eb",
            linewidth=2,
            linestyle="--",
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
        if history_quarters is not None:
            axis.set_xlim(data_index[0], forecast_index[-1])
            axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            axis.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _: (
                        f"{mdates.num2date(value).year}Q"
                        f"{(mdates.num2date(value).month - 1) // 3 + 1}"
                    )
                )
            )
            plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat[len(PLOTS):]:
        axis.set_visible(False)

    title = "Bayesian VAR: Canadian growth and inflation outlook"
    if history_quarters is not None:
        title += f" (last {history_quarters} quarters plus forecast)"
    figure.suptitle(title, fontsize=15)
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


def create_policy_scenario_chart(
    scenarios: dict[float, dict[str, pd.DataFrame]],
    data: pd.DataFrame,
    output: Path,
) -> None:
    """Save recent history and quarterly scenario paths."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    colors = {2.25: "#2563eb", 2.5: "#059669", 2.75: "#dc2626"}
    history = data.iloc[-4:]
    history_periods = history.index.to_timestamp()
    historical_series = (
        ("ca_gdp_growth", axes[0, 0]),
        ("headline_inflation", axes[0, 1]),
        ("core_inflation", axes[1, 0]),
        ("overnight_rate", axes[1, 1]),
    )
    for name, axis in historical_series:
        axis.plot(
            history_periods,
            history[name],
            color="#1f2937",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Historical",
        )
    for rate, paths in scenarios.items():
        central = paths["central"]
        low = paths["low"]
        high = paths["high"]
        periods = central.index.to_timestamp()
        connected_periods = periods.insert(0, history_periods[-1])
        axes[0, 0].plot(
            connected_periods,
            [history["ca_gdp_growth"].iloc[-1], *central["aggregate_growth"]],
            marker="o",
            markersize=4,
            linestyle="--",
            color=colors[rate],
            label=f"{rate:.2f}% forecast",
        )
        axes[0, 0].fill_between(
            periods,
            low["aggregate_growth"],
            high["aggregate_growth"],
            color=colors[rate],
            alpha=0.08,
        )
        axes[0, 1].plot(
            connected_periods,
            [history["headline_inflation"].iloc[-1], *central["headline_inflation"]],
            marker="o",
            markersize=4,
            linestyle="--",
            color=colors[rate],
            label=f"{rate:.2f}% forecast",
        )
        axes[1, 0].plot(
            connected_periods,
            [history["core_inflation"].iloc[-1], *central["core_inflation"]],
            marker="o",
            markersize=4,
            linestyle="--",
            color=colors[rate],
            label=f"{rate:.2f}% forecast",
        )
        axes[1, 1].plot(
            connected_periods,
            [history["overnight_rate"].iloc[-1], *central["overnight_rate"]],
            marker="o",
            markersize=4,
            linestyle="--",
            color=colors[rate],
            label=f"{rate:.2f}% forecast",
        )
    axes[0, 0].set_title("Canadian GDP growth by policy scenario")
    axes[0, 0].set_ylabel("Percent, annualized")
    axes[0, 0].set_ylim(0, 4)
    axes[0, 1].set_title("Headline inflation")
    axes[0, 1].set_ylabel("Percent, year over year")
    axes[1, 0].set_title("Core inflation proxy")
    axes[1, 0].set_ylabel("Percent, year over year")
    axes[1, 1].set_title("Overnight rate")
    axes[1, 1].set_ylabel("Percent")
    for axis in axes.flat:
        axis.set_xlabel("Forecast quarter")
        axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: (
                    f"{mdates.num2date(value).year}Q"
                    f"{(mdates.num2date(value).month - 1) // 3 + 1}"
                )
            )
        )
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(title="Overnight rate", frameon=False, fontsize=8)
        axis.axvline(
            history_periods[-1],
            color="#6b7280",
            linestyle="--",
            linewidth=1,
        )
    figure.suptitle("Policy-rate scenarios: GDP, inflation, and overnight rate", fontsize=15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_policy_scenarios(
    data: pd.DataFrame,
    component_levels: pd.DataFrame,
    horizon: int = 8,
    draws: int = 2_000,
) -> tuple[dict[float, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Run 2.25%, 2.50%, and 2.75% policy paths through both modules."""
    component_levels, data = component_levels.align(data, join="inner", axis=0)
    scenarios: dict[float, dict[str, pd.DataFrame]] = {}
    summary_rows: list[dict[str, float | str]] = []
    for rate in (2.25, 2.50, 2.75):
        policy_path = [rate] * horizon
        bvar = BayesianVAR(BVARConfig(draws=draws)).fit(
            data,
            component_levels=component_levels,
        )
        bvar_forecast = bvar.forecast(
            horizon=horizon,
            future_paths={"overnight_rate": policy_path},
        )
        paths: dict[str, pd.DataFrame] = {}
        for risk in ("central", "low", "high"):
            components = bvar.forecast_gdp_components(bvar_forecast, risk=risk)
            components["aggregate_growth"] = bvar_forecast[
                {
                    "central": "ca_gdp_growth_mean",
                    "low": "ca_gdp_growth_lower80",
                    "high": "ca_gdp_growth_upper80",
                }[risk]
            ].to_numpy()
            components["headline_inflation"] = bvar_forecast[
                {
                    "central": "headline_inflation_mean",
                    "low": "headline_inflation_lower80",
                    "high": "headline_inflation_upper80",
                }[risk]
            ].to_numpy()
            components["core_inflation"] = bvar_forecast[
                {
                    "central": "core_inflation_mean",
                    "low": "core_inflation_lower80",
                    "high": "core_inflation_upper80",
                }[risk]
            ].to_numpy()
            components["overnight_rate"] = policy_path
            components["total_components_growth"] = components[
                [f"{name}_growth" for name in bvar.gdp_components]
            ].mean(axis=1)
            paths[risk] = components
        scenarios[rate] = paths
        central = paths["central"]
        low = paths["low"]
        high = paths["high"]
        for period, row in central.iterrows():
            summary_rows.append(
                {
                    "policy_rate": rate,
                    "period": str(period),
                    "gdp_growth": row["aggregate_growth"],
                    "gdp_growth_low80": low.loc[period, "aggregate_growth"],
                    "gdp_growth_high80": high.loc[period, "aggregate_growth"],
                    "headline_inflation": row["headline_inflation"],
                    "headline_inflation_low80": low.loc[period, "headline_inflation"],
                    "headline_inflation_high80": high.loc[period, "headline_inflation"],
                    "core_inflation": row["core_inflation"],
                    "core_inflation_low80": low.loc[period, "core_inflation"],
                    "core_inflation_high80": high.loc[period, "core_inflation"],
                    "overnight_rate": row["overnight_rate"],
                    "components_growth": row["total_components_growth"],
                    **{
                        f"{name}_growth": row[f"{name}_growth"]
                        for name in bvar.gdp_components
                    },
                }
            )
    return scenarios, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data") / "bvar_quarterly.csv")
    parser.add_argument("--output", type=Path, default=Path("output") / "bvar_forecast.png")
    parser.add_argument(
        "--zoomed-output",
        type=Path,
        default=Path("output") / "bvar_forecast_zoomed.png",
    )
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument(
        "--transmission-output",
        type=Path,
        default=Path("output") / "policy_transmission.csv",
    )
    parser.add_argument(
        "--components-data",
        type=Path,
        default=Path("data") / "gdp_components_quarterly.csv",
    )
    parser.add_argument(
        "--scenario-output",
        type=Path,
        default=Path("output") / "policy_scenarios.csv",
    )
    parser.add_argument(
        "--scenario-chart",
        type=Path,
        default=Path("output") / "policy_scenarios.png",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col="period")
    data.index = pd.PeriodIndex(data.index, freq="Q")
    component_levels = load_component_csv(args.components_data)
    model = BayesianVAR(BVARConfig(draws=args.draws)).fit(
        data,
        component_levels=component_levels,
    )
    args.transmission_output.parent.mkdir(parents=True, exist_ok=True)
    model.policy_transmission_effects().to_csv(
        args.transmission_output,
        index_label="variable",
    )
    forecast = model.forecast(
        horizon=8,
        future_paths={"overnight_rate": [data["overnight_rate"].iloc[-1]] * 8}
    )
    create_forecast_chart(data, forecast, args.output)
    create_forecast_chart(data, forecast, args.zoomed_output, history_quarters=4)
    scenarios, summary = run_policy_scenarios(
        data,
        component_levels,
        draws=args.draws,
    )
    args.scenario_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.scenario_output, index=False)
    create_policy_scenario_chart(scenarios, data, args.scenario_chart)
    print(f"Saved forecast chart to {args.output}")
    print(f"Saved zoomed forecast chart to {args.zoomed_output}")
    print(f"Saved policy scenario numbers to {args.scenario_output}")
    print(f"Saved policy scenario chart to {args.scenario_chart}")
    print(f"Saved policy transmission effects to {args.transmission_output}")


if __name__ == "__main__":
    main()
