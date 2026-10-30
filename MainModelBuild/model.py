"""Bayesian VAR forecasting engine for the Governor's Challenge.

The model expects quarterly, already-transformed data.  A practical default
specification is:

    real_brent, us_gdp_growth, ca_gdp_growth, core_inflation,
    headline_inflation, overnight_rate, ca_exchange_rate

The engine uses a conjugate Bayesian ridge approximation to a Minnesota prior.
This keeps the first reduced-form version transparent and dependency-light.
Structural sign restrictions and conditional scenarios can be added on top of
the fitted reduced-form model without changing the forecasting interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BVARConfig:
    """Hyperparameters for the reduced-form Bayesian VAR."""

    lags: int = 4
    prior_tightness: float = 0.01
    prior_decay: float = 1.0
    draws: int = 2_000
    forecast_horizon: int = 12
    random_seed: int = 42
    random_walk_variables: tuple[str, ...] = ("overnight_rate", "ca_exchange_rate")


class BayesianVAR:
    """Estimate a quarterly Bayesian VAR with a Minnesota-style prior.

    ``fit`` takes a DataFrame whose columns are transformed quarterly series.
    For example, GDP and prices should normally be quarter-over-quarter
    annualized growth rates, while interest rates can remain in percentage
    points.  Missing values are rejected rather than silently imputed.
    """

    def __init__(
        self,
        config: BVARConfig | None = None,
        variables: Sequence[str] | None = None,
    ) -> None:
        self.config = config or BVARConfig()
        if self.config.lags < 1 or self.config.draws < 1:
            raise ValueError("lags and draws must be positive")
        self.variables = list(variables) if variables is not None else [
            "real_brent",
            "us_gdp_growth",
            "ca_gdp_growth",
            "core_inflation",
            "headline_inflation",
            "overnight_rate",
            "ca_exchange_rate",
        ]
        if len(self.variables) == 0:
            raise ValueError("at least one variable is required")
        self._fitted = False

    def fit(self, data: pd.DataFrame) -> "BayesianVAR":
        """Fit the BVAR and return ``self``.

        The input index must be quarterly or otherwise ordered chronologically.
        At least ``lags + 5`` complete observations are required.
        """
        missing = [name for name in self.variables if name not in data.columns]
        if missing:
            raise ValueError(f"missing model columns: {', '.join(missing)}")
        frame = data.loc[:, self.variables].copy()
        if frame.isna().any().any():
            bad = frame.columns[frame.isna().any()].tolist()
            raise ValueError(f"missing values in model columns: {', '.join(bad)}")
        if len(frame) <= self.config.lags + 5:
            raise ValueError("not enough observations for the requested lag length")

        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("model data must contain only finite numeric values")

        y = values[self.config.lags :]
        x = np.column_stack(
            [
                np.ones(len(y)),
                *[
                    values[self.config.lags - lag : -lag]
                    for lag in range(1, self.config.lags + 1)
                ],
            ]
        )
        n_variables = len(self.variables)
        scale = np.std(y, axis=0, ddof=1)
        scale[scale <= 1e-12] = 1.0

        prior_precision = np.zeros((x.shape[1], x.shape[1]))
        prior_mean = np.zeros((x.shape[1], n_variables))
        prior_precision[0, 0] = 1e-6
        for lag in range(1, self.config.lags + 1):
            for variable in range(n_variables):
                position = 1 + (lag - 1) * n_variables + variable
                prior_variance = (
                    self.config.prior_tightness**2
                    / lag ** (2 * self.config.prior_decay)
                )
                prior_precision[position, position] = 1.0 / max(
                    prior_variance, 1e-12
                )
                if (
                    variable < n_variables
                    and lag == 1
                    and self.variables[variable] in self.config.random_walk_variables
                ):
                    prior_mean[position, variable] = 1.0

        # Equation-specific scale adjustment: own and cross-variable lags have
        # prior variance proportional to sigma_i^2 / sigma_j^2.
        prior_precision_by_equation = np.zeros(
            (n_variables, x.shape[1], x.shape[1])
        )
        for equation in range(n_variables):
            precision = prior_precision.copy()
            for lag in range(1, self.config.lags + 1):
                for regressor in range(n_variables):
                    position = 1 + (lag - 1) * n_variables + regressor
                    variance = (
                        self.config.prior_tightness**2
                        / lag ** (2 * self.config.prior_decay)
                        * scale[equation] ** 2
                        / scale[regressor] ** 2
                    )
                    precision[position, position] = 1.0 / max(variance, 1e-12)
            prior_precision_by_equation[equation] = precision

        coefficients = np.zeros((n_variables, x.shape[1]))
        coefficient_covariances = []
        residuals = np.zeros_like(y)
        for equation in range(n_variables):
            precision = prior_precision_by_equation[equation]
            posterior_precision = x.T @ x + precision
            covariance = np.linalg.inv(
                posterior_precision + np.eye(x.shape[1]) * 1e-10
            )
            mean = covariance @ (
                x.T @ y[:, equation] + precision @ prior_mean[:, equation]
            )
            coefficients[equation] = mean
            coefficient_covariances.append(covariance)
            residuals[:, equation] = y[:, equation] - x @ mean

        residual_covariance = residuals.T @ residuals / max(len(y) - x.shape[1], 1)
        residual_covariance += np.eye(n_variables) * 1e-10
        self._coefficients = coefficients
        self._coefficient_covariances = coefficient_covariances
        self._residual_covariance = residual_covariance
        self._history = values
        self._index = frame.index
        self._last_values = values[-self.config.lags :].copy()
        self._fitted = True
        return self

    def forecast(
        self,
        horizon: int | None = None,
        draws: int | None = None,
        future_paths: Mapping[str, Sequence[float]] | None = None,
    ) -> pd.DataFrame:
        """Return posterior forecast means and 80% intervals by quarter.

        ``future_paths`` can condition exogenous-looking variables such as oil
        or the policy rate.  A supplied path must contain exactly ``horizon``
        values.  Conditioning is predictive, not automatically a causal policy
        effect; structural identification belongs in a later extension.
        """
        self._require_fit()
        horizon = self.config.forecast_horizon if horizon is None else horizon
        draws = self.config.draws if draws is None else draws
        if horizon < 1 or draws < 1:
            raise ValueError("horizon and draws must be positive")
        paths = dict(future_paths or {})
        unknown = set(paths) - set(self.variables)
        if unknown:
            raise ValueError(f"future path has unknown columns: {sorted(unknown)}")
        for name, path in paths.items():
            path_values = np.asarray(path, dtype=float)
            if len(path_values) != horizon or not np.isfinite(path_values).all():
                raise ValueError(f"future path for {name} must have {horizon} finite values")
            paths[name] = path_values

        rng = np.random.default_rng(self.config.random_seed)
        simulations = np.empty((draws, horizon, len(self.variables)))
        for draw in range(draws):
            simulated_coefficients = []
            for equation, covariance in enumerate(self._coefficient_covariances):
                covariance = (covariance + covariance.T) / 2
                covariance += np.eye(covariance.shape[0]) * 1e-10
                simulated_coefficients.append(
                    rng.multivariate_normal(self._coefficients[equation], covariance)
                )
            history = self._last_values.copy()
            for step in range(horizon):
                regressors = [1.0]
                for lag in range(1, self.config.lags + 1):
                    regressors.extend(history[-lag])
                prediction = np.array(
                    [
                        np.dot(simulated_coefficients[equation], regressors)
                        for equation in range(len(self.variables))
                    ]
                )
                prediction += rng.multivariate_normal(
                    np.zeros(len(self.variables)), self._residual_covariance
                )
                for name, path in paths.items():
                    prediction[self.variables.index(name)] = path[step]
                simulations[draw, step] = prediction
                history = np.vstack([history, prediction])[-self.config.lags :]

        lower, mean, upper = np.quantile(
            simulations, [0.10, 0.50, 0.90], axis=0
        )
        index = self._future_index(horizon)
        output = pd.DataFrame(index=index)
        for position, name in enumerate(self.variables):
            output[f"{name}_mean"] = mean[:, position]
            output[f"{name}_lower80"] = lower[:, position]
            output[f"{name}_upper80"] = upper[:, position]
        output.index.name = self._index.name or "period"
        return output

    def annual_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Aggregate quarterly forecast output into calendar-year averages."""
        means = forecast.filter(regex="_mean$")
        if isinstance(forecast.index, (pd.PeriodIndex, pd.DatetimeIndex)):
            years = forecast.index.year
        else:
            raise ValueError("forecast must have a quarterly PeriodIndex or date index")
        annual = means.groupby(years).mean()
        annual.index.name = "year"
        return annual

    def _future_index(self, horizon: int) -> pd.Index:
        last = self._index[-1]
        if isinstance(last, pd.Period):
            return pd.period_range(last + 1, periods=horizon, freq=last.freq)
        if isinstance(last, pd.Timestamp):
            return pd.period_range(last.to_period("Q") + 1, periods=horizon, freq="Q")
        return pd.RangeIndex(start=len(self._history), stop=len(self._history) + horizon)

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit the model before requesting a forecast")


def load_quarterly_csv(
    path: str,
    date_column: str = "date",
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and validate a quarterly model dataset from CSV."""
    data = pd.read_csv(path, parse_dates=[date_column]).set_index(date_column)
    if variables is not None:
        data = data.loc[:, list(variables)]
    return data.sort_index()


if __name__ == "__main__":
    periods = pd.period_range("2000Q1", periods=100, freq="Q")
    rng = np.random.default_rng(7)
    example = pd.DataFrame(
        rng.normal(size=(len(periods), 7)),
        index=periods,
        columns=BayesianVAR().variables,
    )
    model = BayesianVAR().fit(example)
    print(model.forecast(horizon=12).round(3))