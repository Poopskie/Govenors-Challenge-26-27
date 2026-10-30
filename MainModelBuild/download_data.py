"""Download and transform public macroeconomic data for the Bayesian VAR.

The dataset combines FRED, Bank of Canada Valet, and Statistics Canada table
18-10-0004-01.  Headline CPI is total CPI; core CPI is Statistics Canada's
"all-items excluding food and energy" measure.  The latter is a core-CPI
measure, but it is not one of the Bank of Canada's preferred CPI-common,
CPI-median, or CPI-trim measures.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
VALET_URL = (
    "https://www.bankofcanada.ca/valet/observations/{series}/json"
    "?start_date=2000-01-01"
)
STATCAN_CPI_URL = (
    "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip"
)
START_DATE = "2000-01-01"

# Statistics Canada table 18-10-0004-01, Canada, monthly:
# "All-items excluding food and energy" is a transparent core-CPI measure.
CORE_CPI_VECTOR = "v41691773"


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "GovernorChallengeBVAR/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def fred_series(series: str) -> pd.Series:
    raw = pd.read_csv(io.BytesIO(_read_url(FRED_URL.format(series=series))))
    dates = pd.to_datetime(raw.iloc[:, 0])
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    return pd.Series(values.to_numpy(), index=dates, name=series).dropna()


def statcan_cpi_series(vector: str) -> pd.Series:
    """Download a Statistics Canada CPI vector from table 18-10-0004-01."""
    raw = _read_url(STATCAN_CPI_URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        with archive.open("18100004.csv") as csv_file:
            frame = pd.read_csv(
                csv_file,
                usecols=["REF_DATE", "VECTOR", "VALUE"],
                dtype={"VECTOR": "string"},
            )
    selected = frame.loc[frame["VECTOR"].str.lower() == vector.lower()].copy()
    if selected.empty:
        raise ValueError(f"Statistics Canada CPI vector not found: {vector}")
    dates = pd.to_datetime(selected["REF_DATE"])
    values = pd.to_numeric(selected["VALUE"], errors="coerce")
    return pd.Series(values.to_numpy(), index=dates, name=vector).dropna()


def valet_series(series: str) -> pd.Series:
    payload = json.loads(_read_url(VALET_URL.format(series=series)))
    observations = payload["observations"]
    dates = pd.to_datetime([item["d"] for item in observations])
    values = [item[series]["v"] for item in observations]
    return pd.Series(pd.to_numeric(values, errors="coerce"), index=dates, name=series).dropna()


def quarterly_growth(series: pd.Series) -> pd.Series:
    quarterly = series.resample("QE").mean()
    return 400 * np.log(quarterly).diff()


def quarterly_year_over_year(series: pd.Series) -> pd.Series:
    quarterly = series.resample("QE").mean()
    return 100 * quarterly.pct_change(4)


def build_dataset() -> pd.DataFrame:
    us_gdp = fred_series("GDPC1")
    ca_gdp_growth = fred_series("CANGDPRQPSMEI")
    oil = fred_series("DCOILBRENTEU")
    cpi = valet_series("V41690973")
    # These FRED series extend the model history beyond the Bank's current
    # Valet series: CAD per USD (daily) and the Canadian central-bank rate
    # (monthly), both with observations well before 2017.
    exchange = fred_series("DEXCAUS")
    policy_rate = fred_series("IRSTCI01CAM156N")

    headline = quarterly_year_over_year(cpi).rename("headline_inflation")
    core = quarterly_year_over_year(
        statcan_cpi_series(CORE_CPI_VECTOR)
    ).rename("core_inflation")

    data = pd.concat(
        [
            quarterly_growth(oil).rename("real_brent"),
            quarterly_growth(us_gdp).rename("us_gdp_growth"),
            ca_gdp_growth.resample("QE").mean().rename("ca_gdp_growth"),
            core,
            headline,
            policy_rate.resample("QE").mean().rename("overnight_rate"),
            exchange.resample("QE").mean().rename("ca_exchange_rate"),
        ],
        axis=1,
        sort=False,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data.index >= pd.Timestamp(START_DATE)]
    data.index = data.index.to_period("Q")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "bvar_quarterly.csv",
        help="destination CSV path",
    )
    args = parser.parse_args()
    data = build_dataset()
    if len(data) <= 20:
        raise RuntimeError(f"only {len(data)} complete quarters available; need more history")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index_label="period")
    print(f"Saved {len(data)} quarters from {data.index[0]} through {data.index[-1]} to {args.output}")


if __name__ == "__main__":
    main()
