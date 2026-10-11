"""Load, validate and prepare a long-format weekly panel.

One code path for the demo CSV and for a customer upload (spec Section 6), so the
API's BYOD flow and the demo fit go through exactly the same preparation.

Required columns: week, units, price. Optional: city, segment, promo, feature,
holdout. Anything missing gets a safe default and sets a flag, and the flags are
what make the model build itself from the columns that are actually present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["week", "units", "price"]
OPTIONAL = ["city", "segment", "promo", "feature", "holdout"]
TRAIT_CANDIDATES = ["log_population", "median_income_k", "share_age_18_34"]
MIX_PREFIX = "mix_"  # cities file: mix_<segment slug> = that segment's share of the city

SEASON_PERIOD = 52.0
N_FOURIER = 2


class DataError(ValueError):
    """The upload cannot be fitted at all; the message is shown to the user."""


@dataclass
class ModelData:
    """Everything the PyMC model and the exporter need, already indexed."""

    cities: list[str]
    segments: list[str]
    n_weeks: int

    # one entry per observation row
    city_idx: np.ndarray
    seg_idx: np.ndarray
    t: np.ndarray
    units: np.ndarray
    price: np.ndarray
    promo: np.ndarray
    promo_last_week: np.ndarray
    feature: np.ndarray
    week_start: list[str]

    shares: np.ndarray            # [C][S], rows sum to 1
    log_share: np.ndarray         # per row, the offset
    season_design: np.ndarray     # [n_rows][2*N_FOURIER]

    base_price: float
    unit_cost: float
    traits_raw: np.ndarray        # [C][P]
    traits_z: np.ndarray          # [C][P]
    trait_names: list[str]
    trait_mean: np.ndarray
    trait_std: np.ndarray

    has_promo: bool
    has_feature: bool
    has_traits: bool
    has_segments: bool
    has_holdout: bool
    shares_source: str = "volume"  # "mix" when the cities file supplied the customer mix
    dataset_name: str = "dataset"
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.units)

    @property
    def n_cities(self) -> int:
        return len(self.cities)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_traits(self) -> int:
        return len(self.trait_names)

    def subset(self, mask: np.ndarray) -> ModelData:
        """Row subset with the same indexing — used for the time holdout refit."""
        mask = np.asarray(mask, dtype=bool)
        kw = dict(self.__dict__)
        for key in (
            "city_idx", "seg_idx", "t", "units", "price", "promo",
            "promo_last_week", "feature", "log_share",
        ):
            kw[key] = kw[key][mask]
        kw["season_design"] = self.season_design[mask]
        kw["week_start"] = [w for w, keep in zip(self.week_start, mask, strict=True) if keep]
        return ModelData(**kw)


def season_design(t: np.ndarray, n_fourier: int = N_FOURIER, period: float = SEASON_PERIOD):
    """Fourier design matrix. Must match contract/README.md and web/lib/simulate.ts."""
    t = np.asarray(t, dtype=float)
    cols = []
    for k in range(1, n_fourier + 1):
        ang = 2.0 * np.pi * k * t / period
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def _mix_columns(cities: pd.DataFrame | None, segments: list[str]) -> list[str] | None:
    """The cities-file columns holding the customer mix, if every segment has one."""
    if cities is None or not len(cities):
        return None
    cols = {c.strip().lower() for c in cities.columns}
    wanted = [MIX_PREFIX + s.strip().lower().replace(" ", "_") for s in segments]
    return wanted if all(w in cols for w in wanted) else None


def _segment_shares(
    df: pd.DataFrame,
    city_names: list[str],
    seg_names: list[str],
    cities: pd.DataFrame | None,
) -> tuple[np.ndarray, str]:
    """[C][S] shares with rows summing to 1, from the cities file if it has them."""
    wanted = _mix_columns(cities, seg_names)
    if wanted is not None:
        cdf = cities.rename(columns={c: c.strip().lower() for c in cities.columns})
        if "city" in cdf.columns:
            cdf = cdf.set_index("city").reindex(city_names)
            mix = cdf[wanted].to_numpy(dtype=float)
            if not np.isnan(mix).any() and (mix > 0).all():
                return mix / mix.sum(axis=1, keepdims=True), "mix"

    vol = df.pivot_table(
        index="city_idx", columns="seg_idx", values="units", aggfunc="sum", fill_value=0.0
    ).reindex(index=range(len(city_names)), columns=range(len(seg_names)), fill_value=0.0)
    shares = vol.to_numpy(dtype=float)
    row_tot = shares.sum(axis=1, keepdims=True)
    shares = np.where(row_tot > 0, shares / np.maximum(row_tot, 1e-12), 1.0 / len(seg_names))
    return shares, "volume"


def validate(df: pd.DataFrame, cities: pd.DataFrame | None = None) -> dict:
    """Validation report for the /data page. Never raises on a fixable problem."""
    cols = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in cols]
    report: dict = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required": missing,
        "ok": not missing,
        "errors": [],
        "warnings": [],
        "info": [],
    }
    if missing:
        report["errors"].append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Ripple needs at least {', '.join(REQUIRED)}."
        )
        return report

    units = pd.to_numeric(df[cols["units"]], errors="coerce")
    price = pd.to_numeric(df[cols["price"]], errors="coerce")
    week = pd.to_datetime(df[cols["week"]], errors="coerce")

    n_bad_week = int(week.isna().sum())
    n_bad_units = int(units.isna().sum())
    n_neg = int((units < 0).sum())
    n_bad_price = int((price.isna() | (price <= 0)).sum())

    report["weeks"] = int(week.dropna().nunique())
    report["week_range"] = (
        [str(week.min().date()), str(week.max().date())] if week.notna().any() else None
    )
    report["cities"] = int(df[cols["city"]].nunique()) if "city" in cols else 1
    report["segments"] = int(df[cols["segment"]].nunique()) if "segment" in cols else 1
    report["missing_values"] = int(df.isna().sum().sum())
    report["negative_units"] = n_neg
    report["price_cv"] = (
        float(price.std() / price.mean()) if price.notna().any() and price.mean() else 0.0
    )

    if n_bad_week:
        report["errors"].append(f"{n_bad_week} row(s) have a week that is not a date.")
    if n_bad_units:
        report["errors"].append(f"{n_bad_units} row(s) have units that are not a number.")
    if n_neg:
        report["errors"].append(f"{n_neg} row(s) have negative units. Units must be a count.")
    if n_bad_price:
        report["errors"].append(f"{n_bad_price} row(s) have a price that is zero or missing.")
    if report["weeks"] < 20:
        report["warnings"].append(
            f"Only {report['weeks']} weeks of data. Seasonality and the promo dip need "
            "more history; the intervals will be wide."
        )
    if report["price_cv"] < 0.01:
        report["warnings"].append(
            "Price never changes — elasticity cannot be learned from this data. "
            "Add weeks with different prices to run price scenarios."
        )
    if "promo" not in cols:
        report["info"].append("No promo column: promotion scenarios will be switched off.")
    if "segment" not in cols:
        report["info"].append("No segment column: everything is one group called 'All'.")
    if cities is None:
        report["info"].append(
            "No cities file: new-city scenarios need city traits "
            f"({', '.join(TRAIT_CANDIDATES)})."
        )
    else:
        lower = [x.strip().lower() for x in cities.columns]
        found = [c for c in TRAIT_CANDIDATES if c in lower]
        report["info"].append(f"City traits found: {', '.join(found) if found else 'none'}.")
        if any(c.startswith(MIX_PREFIX) for c in lower):
            report["info"].append("Customer mix found (mix_* columns) and used as the offset.")
        else:
            report["info"].append(
                "No mix_* columns: the customer mix is derived from observed volume."
            )

    report["ok"] = not report["errors"]
    return report


def prepare(
    df: pd.DataFrame,
    cities: pd.DataFrame | None = None,
    *,
    base_price: float | None = None,
    unit_cost: float = 0.0,
    dataset_name: str = "dataset",
) -> ModelData:
    """Long CSV -> ModelData. Raises DataError only when the data cannot be fitted."""
    rep = validate(df, cities)
    if rep["errors"]:
        raise DataError(" ".join(rep["errors"]))

    df = df.rename(columns={c: c.strip().lower() for c in df.columns}).copy()
    df["week"] = pd.to_datetime(df["week"])
    df["units"] = pd.to_numeric(df["units"])
    df["price"] = pd.to_numeric(df["price"])

    has_segments = "segment" in df.columns and df["segment"].nunique() > 1
    has_promo = "promo" in df.columns
    has_feature = "feature" in df.columns
    has_holdout = "holdout" in df.columns

    if "city" not in df.columns:
        df["city"] = "All"
    if "segment" not in df.columns:
        df["segment"] = "All"
    for col in ("promo", "feature"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col]).fillna(0.0).clip(0, 1)

    city_names = sorted(df["city"].unique().tolist())
    seg_names = sorted(df["segment"].unique().tolist())
    weeks_sorted = sorted(df["week"].unique())
    week_to_t = {w: i for i, w in enumerate(weeks_sorted)}

    df = df.sort_values(["week", "city", "segment"]).reset_index(drop=True)
    df["t"] = df["week"].map(week_to_t).astype(int)
    df["city_idx"] = df["city"].map({c: i for i, c in enumerate(city_names)}).astype(int)
    df["seg_idx"] = df["segment"].map({s: i for i, s in enumerate(seg_names)}).astype(int)

    # promo_last_week per city x segment, by week order — this is what identifies b_dip
    df = df.sort_values(["city_idx", "seg_idx", "t"]).reset_index(drop=True)
    df["promo_last_week"] = (
        df.groupby(["city_idx", "seg_idx"])["promo"].shift(1).fillna(0.0).to_numpy()
    )
    df = df.sort_values(["t", "city_idx", "seg_idx"]).reset_index(drop=True)

    # The share offset is what lets a new city with a different customer mix be simulated
    # without refitting. Prefer the customer mix declared in the cities file: it is a
    # property of the city. Deriving it from observed volume instead would fold price and
    # promo exposure into the offset, and a_seg / a_city would no longer mean what they say.
    shares, shares_source = _segment_shares(df, city_names, seg_names, cities)

    # traits
    trait_names: list[str] = []
    traits_raw = np.zeros((len(city_names), 0))
    if cities is not None and len(cities):
        cdf = cities.rename(columns={c: c.strip().lower() for c in cities.columns}).copy()
        if "city" in cdf.columns:
            trait_names = [c for c in TRAIT_CANDIDATES if c in cdf.columns]
            if trait_names:
                cdf = cdf.set_index("city").reindex(city_names)
                if cdf[trait_names].isna().any().any():
                    trait_names = []  # a city has no traits -> cannot use traits at all
                else:
                    traits_raw = cdf[trait_names].to_numpy(dtype=float)
    has_traits = bool(trait_names)
    if has_traits:
        trait_mean = traits_raw.mean(axis=0)
        sd = traits_raw.std(axis=0, ddof=0)
        trait_std = np.where(sd > 0, sd, 1.0)
        traits_z = (traits_raw - trait_mean) / trait_std
    else:
        trait_mean = np.zeros(0)
        trait_std = np.ones(0)
        traits_z = np.zeros((len(city_names), 0))

    if base_price:
        bp = float(base_price)
    elif (df["promo"] == 0).any():
        # the typical non-promo price is the natural reference for the elasticity term
        bp = float(np.median(df.loc[df["promo"] == 0, "price"]))
    else:
        bp = float(np.median(df["price"]))

    return ModelData(
        cities=city_names,
        segments=seg_names,
        n_weeks=len(weeks_sorted),
        city_idx=df["city_idx"].to_numpy(),
        seg_idx=df["seg_idx"].to_numpy(),
        t=df["t"].to_numpy(),
        units=df["units"].to_numpy(dtype=float),
        price=df["price"].to_numpy(dtype=float),
        promo=df["promo"].to_numpy(dtype=float) if has_promo else np.zeros(len(df)),
        promo_last_week=(
            df["promo_last_week"].to_numpy(dtype=float) if has_promo else np.zeros(len(df))
        ),
        feature=df["feature"].to_numpy(dtype=float) if has_feature else np.zeros(len(df)),
        week_start=[str(pd.Timestamp(w).date()) for w in df["week"]],
        shares=shares,
        log_share=np.log(shares[df["city_idx"].to_numpy(), df["seg_idx"].to_numpy()]),
        season_design=season_design(df["t"].to_numpy()),
        base_price=bp,
        unit_cost=float(unit_cost),
        traits_raw=traits_raw,
        traits_z=traits_z,
        trait_names=trait_names,
        trait_mean=trait_mean,
        trait_std=trait_std,
        has_promo=has_promo,
        has_feature=has_feature,
        has_traits=has_traits,
        has_segments=has_segments,
        has_holdout=has_holdout,
        shares_source=shares_source,
        dataset_name=dataset_name,
        warnings=list(rep["warnings"]),
    )


def load_demo(unit_cost: float = 9.0) -> ModelData:
    """The Brightside demo dataset, loaded exactly like a customer upload."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    df = pd.read_csv(data_dir / "brightside_weekly.csv")
    cities = pd.read_csv(data_dir / "brightside_cities.csv")
    return prepare(
        df,
        cities,
        base_price=24.0,
        unit_cost=unit_cost,
        dataset_name="Demo · Brightside Coffee Subscriptions",
    )
