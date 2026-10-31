"""Quarterly Canadian GDP expenditure-components model.

The component data are real, seasonally adjusted, annual-rate values from
Statistics Canada table 36-10-0104-01.  The model is intentionally transparent: in integrated mode each component's
growth depends on its own lag and the BVAR's forecast drivers. Aggregate GDP
is reconstructed from the forecast component levels and returned to the BVAR.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Sequence
import urllib.request

import numpy as np
import pandas as pd


STATCAN_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/36100104-eng.zip"
COMPONENTS = (
    "consumption",
    "residential_investment",
    "nonresidential_investment",
    "government_spending",
    "exports",
    "imports",
)
ESTIMATE_NAMES = {
    "consumption": "Household final consumption expenditure",
    "residential_investment": "Residential structures",
    "nonresidential_investment": "Non-residential structures, machinery and equipment",
    "government_spending": "General governments final consumption expenditure",
    "exports": "Exports of goods and services",
    "imports": "Less: imports of goods and services",
}


def download_component_data(
    output: str | Path,
    start: str = "2000-01",
) -> pd.DataFrame:
    """Download and save the real quarterly component levels."""
    request = urllib.request.Request(
        STATCAN_URL,
        headers={"User-Agent": "GovernorChallengeBVAR/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open("36100104.csv") as csv_file:
            raw = pd.read_csv(
                csv_file,
                usecols=[
                    "REF_DATE",
                    "Prices",
                    "Seasonal adjustment",
                    "Estimates",
                    "UOM",
                    "VALUE",
                ],
            )
    selected = raw.loc[
        (raw["Prices"] == "Chained (2017) dollars")
        & (raw["Seasonal adjustment"] == "Seasonally adjusted at annual rates")
        & (raw["UOM"] == "Dollars")
        & raw["Estimates"].isin(ESTIMATE_NAMES.values())
    ].copy()
    selected["component"] = selected["Estimates"].map(
        {value: key for key, value in ESTIMATE_NAMES.items()}
    )
    frame = (
        selected.pivot(index="REF_DATE", columns="component", values="VALUE")
        .sort_index()
        .loc[lambda data: data.index >= start]
        .dropna()
    )
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    frame = frame.loc[:, list(COMPONENTS)].astype(float)
    if frame.empty:
        raise ValueError("Statistics Canada returned no complete component quarters")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index_label="period")
    return frame


class GDPComponentsModel:
    """Forecast real GDP from expenditure components and macroeconomic drivers."""

    def __init__(
        self,
        components: Sequence[str] = COMPONENTS,
        ridge: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        self.components = list(components)
        if not self.components or any(name not in COMPONENTS for name in self.components):
            raise ValueError("components must be selected from the published component names")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = ridge
        self.random_seed = random_seed
        self._fitted = False

    def fit(
        self,
        levels: pd.DataFrame,
        aggregate_growth: pd.Series | None = None,
        policy_rate: pd.Series | None = None,
        drivers: pd.DataFrame | None = None,
    ) -> "GDPComponentsModel":
        """Fit component growth equations to aligned quarterly observations.

        ``drivers`` is the integrated-model path.  In that mode every
        non-GDP BVAR variable is used as a driver, and GDP is reconstructed
        from the forecast component levels.  The aggregate-growth arguments
        remain supported for standalone use.
        """
        missing = set(self.components) - set(levels.columns)
        if missing:
            raise ValueError(f"missing component columns: {sorted(missing)}")
        if drivers is not None:
            if drivers.empty:
                raise ValueError("drivers must contain at least one column")
            frame = pd.concat(
                [
                    np.log(levels.loc[:, self.components])
                    .diff()
                    .mul(400)
                    .add_suffix("_growth"),
                    drivers,
                ],
                axis=1,
            ).dropna()
            driver_names = list(drivers.columns)
        else:
            if aggregate_growth is None or policy_rate is None:
                raise ValueError(
                    "aggregate_growth and policy_rate are required without drivers"
                )
            frame = pd.concat(
                [
                    np.log(levels.loc[:, self.components])
                    .diff()
                    .mul(400)
                    .add_suffix("_growth"),
                    aggregate_growth.rename("aggregate_growth"),
                    policy_rate.rename("policy_rate"),
                ],
                axis=1,
            ).dropna()
            driver_names = ["aggregate_growth", "policy_rate"]
        if "overnight_rate" in frame:
            frame["policy_shock"] = frame["overnight_rate"].diff().fillna(0.0)
        elif "policy_rate" in frame:
            frame["policy_shock"] = frame["policy_rate"].diff().fillna(0.0)
        self._has_policy_shock = "policy_shock" in frame
        if len(frame) < 20:
            raise ValueError("at least 20 complete quarters are required")
        self._coefficients: dict[str, np.ndarray] = {}
        self._residual_std: dict[str, float] = {}
        self._driver_names = driver_names
        self._integrated = drivers is not None
        for component in self.components:
            target = frame[f"{component}_growth"].to_numpy()
            regressors = [
                np.ones(len(frame)),
                frame[f"{component}_growth"].shift(1).fillna(0).to_numpy(),
            ]
            regressors.extend(
                (frame[name] - frame[name].mean()).to_numpy()
                for name in driver_names
            )
            if "overnight_rate" in frame:
                regressors.append(
                    (frame["overnight_rate"] - frame["overnight_rate"].mean())
                    .shift(1)
                    .fillna(0.0)
                    .to_numpy()
                )
            elif "policy_rate" in frame:
                regressors.append(
                    (frame["policy_rate"] - frame["policy_rate"].mean())
                    .shift(1)
                    .fillna(0.0)
                    .to_numpy()
                )
            if "policy_shock" in frame:
                regressors.append(frame["policy_shock"].to_numpy())
            regressors = np.column_stack(regressors)
            penalty = np.eye(regressors.shape[1]) * self.ridge
            penalty[0, 0] = 1e-8
            coefficients = np.linalg.solve(
                regressors.T @ regressors + penalty,
                regressors.T @ target,
            )
            residual = target - regressors @ coefficients
            self._coefficients[component] = coefficients
            self._residual_std[component] = float(np.std(residual, ddof=1))
        self._last_growth = np.log(levels.loc[:, self.components]).diff().mul(400).iloc[-1]
        self._last_levels = levels.loc[:, self.components].iloc[-1].copy()
        self._policy_name = (
            "overnight_rate" if "overnight_rate" in frame else "policy_rate"
        )
        self._driver_means = frame.loc[:, driver_names].mean()
        self._last_drivers = frame.loc[:, driver_names].iloc[-1]
        self._last_policy = float(frame[self._policy_name].iloc[-1])
        self._last_period = levels.index[-1]
        self._fitted = True
        return self

    def forecast(
        self,
        aggregate_growth: Sequence[float] | None = None,
        policy_rate: Sequence[float] | None = None,
        risk: str = "central",
        driver_paths: pd.DataFrame | None = None,
        policy_sensitivity: float = 0.0,
    ) -> pd.DataFrame:
        """Forecast component growth and normalized levels for supplied paths."""
        self._require_fit()
        if policy_sensitivity < 0:
            raise ValueError("policy_sensitivity must be non-negative")
        if self._integrated:
            if driver_paths is None:
                raise ValueError("driver_paths are required for an integrated GDP model")
            missing = set(self._driver_names) - set(driver_paths.columns)
            if missing:
                raise ValueError(f"missing driver paths: {sorted(missing)}")
            paths = driver_paths.loc[:, self._driver_names].copy()
            if len(paths) < 1 or not np.isfinite(paths.to_numpy(dtype=float)).all():
                raise ValueError("driver paths must contain finite values")
        else:
            if aggregate_growth is None or policy_rate is None:
                raise ValueError(
                    "aggregate_growth and policy_rate are required for standalone forecasting"
                )
            aggregate = np.asarray(aggregate_growth, dtype=float)
            policy = np.asarray(policy_rate, dtype=float)
            if len(aggregate) != len(policy) or len(aggregate) < 1:
                raise ValueError("aggregate_growth and policy_rate must have equal non-zero lengths")
            if not np.isfinite(aggregate).all() or not np.isfinite(policy).all():
                raise ValueError("forecast paths must contain only finite values")
            paths = pd.DataFrame(
                {
                    "aggregate_growth": aggregate,
                    "policy_rate": policy,
                }
            )
        if risk not in {"central", "low", "high"}:
            raise ValueError("risk must be central, low, or high")
        growth = np.empty((len(paths), len(self.components)))
        levels = np.empty_like(growth)
        previous = self._last_growth.loc[self.components].to_numpy(dtype=float)
        last_levels = self._last_levels.loc[self.components].to_numpy(dtype=float)
        risk_sign = {"central": 0.0, "low": -1.0, "high": 1.0}[risk]
        for step in range(len(paths)):
            for position, component in enumerate(self.components):
                coefficients = self._coefficients[component]
                regressors = [1.0, previous[position]]
                regressors.extend(
                    paths.iloc[step][name] - self._driver_means[name]
                    for name in self._driver_names
                )
                previous_policy = (
                    self._last_policy
                    if step == 0
                    else float(paths.iloc[step - 1][self._policy_name])
                )
                regressors.append(previous_policy - self._driver_means[self._policy_name])
                if self._has_policy_shock:
                    regressors.append(
                        float(paths.iloc[step][self._policy_name]) - previous_policy
                    )
                fitted_growth = np.asarray(regressors) @ coefficients
                if policy_sensitivity:
                    policy_driver_position = 2 + self._driver_names.index(
                        self._policy_name
                    )
                    fitted_growth -= (
                        regressors[policy_driver_position]
                        * coefficients[policy_driver_position]
                    )
                    fitted_growth -= (
                        regressors[2 + len(self._driver_names)]
                        * coefficients[2 + len(self._driver_names)]
                    )
                    if self._has_policy_shock:
                        fitted_growth -= regressors[-1] * coefficients[-1]
                    fitted_growth -= policy_sensitivity * (
                        float(paths.iloc[step][self._policy_name])
                        - self._last_policy
                    )
                growth[step, position] = (
                    fitted_growth
                    + risk_sign * self._residual_std[component]
                    * 0.5
                )
            levels[step] = last_levels * np.exp(growth[step] / 400)
            last_levels = levels[step]
            previous = growth[step]
        index = (
            paths.index
            if isinstance(paths.index, pd.PeriodIndex)
            else pd.period_range(self._last_period + 1, periods=len(paths), freq="Q")
        )
        result = pd.DataFrame(index=index)
        for position, component in enumerate(self.components):
            result[f"{component}_growth"] = growth[:, position]
            result[f"{component}_index"] = 100 * levels[:, position] / self._last_levels.loc[component]
        if self._integrated:
            previous_total = (
                self._last_levels.sum() - self._last_levels["imports"]
            )
            total_levels = levels.sum(axis=1) - levels[:, self.components.index("imports")]
            result["aggregate_growth"] = 400 * np.log(
                total_levels / np.r_[previous_total, total_levels[:-1]]
            )
        result.index.name = "period"
        return result

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit the GDP components model before forecasting")


def load_component_csv(path: str | Path) -> pd.DataFrame:
    """Load a component-level CSV created by :func:`download_component_data`."""
    frame = pd.read_csv(path, index_col="period")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return frame.sort_index().loc[:, list(COMPONENTS)]


if __name__ == "__main__":
    download_component_data(Path("data") / "gdp_components_quarterly.csv")