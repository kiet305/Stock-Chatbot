import pandas as pd
from dagster import asset, AssetIn, Nothing, StaticPartitionsDefinition, Output
from datetime import timezone, timedelta
from datetime import date, datetime
import re

def extract_issued_shares(title: str) -> float | None:
    if pd.isna(title):
        return None
    # tìm cụm số có dấu . hoặc ,
    match = re.search(r"([\d\.,]+)", title)
    if not match:
        return None
    # bỏ dấu phân cách hàng nghìn
    number_str = match.group(1).replace(".", "").replace(",", "")
    try:
        return float(number_str)
    except ValueError:
        return None
    
import numpy as np

def safe_divide(numer, denom):
    return np.where(
        (denom > 0) & pd.notna(denom),
        numer / denom,
        np.nan,
    )
    
def date_to_quarter(d, delay: int = 0):
    dt = pd.to_datetime(d)
    q = (dt.month - 1) // 3 + 1 - delay
    y = dt.year
    if q == 0:
        q = 4
        y = y -1
    return y, q

def calc_trailing(s: pd.Series, period: int = 4) -> pd.Series:
    rolling_sum = s.rolling(period, min_periods=period).sum()
    expanding_mean = s.expanding(1).mean() * period
    return rolling_sum.fillna(expanding_mean)

def build_wide_financials(reports: pd.DataFrame) -> pd.DataFrame:
    df = reports[
        reports["criteria"].isin(
            [
                "profit",
                "equity",
                "total_assets",
                "revenue",
                "interest_income",
                "interest_expenses",
                "deposit_at_SBV",
                "deposit_at_FI",
                "investment_securities",
                "customer_loan",
            ]
        )
    ].copy()

    wide = (
        df.pivot_table(
            index=["ticker", "year", "quarter"],
            columns="criteria",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["ticker", "year", "quarter"])
    )

    # ===== ADD earning assets =====
    EARNING_COLS = [
        "deposit_at_SBV",
        "deposit_at_FI",
        "investment_securities",
        "customer_loan",
    ]

    wide[EARNING_COLS] = wide[EARNING_COLS].fillna(0)
    wide["earning_assets"] = wide[EARNING_COLS].sum(axis=1)

    return wide

def calc_roe_roa_with_trailing(
    df: pd.DataFrame,
    period: int = 2,
) -> pd.DataFrame:

    df = df.copy()
    # Equity & Assets trailing average 2 quý
    df["avg_equity"] = (
        df.groupby("ticker")["equity"]
        .transform(lambda s: calc_trailing(s, period)) / period
    )

    df["avg_assets"] = (
        df.groupby("ticker")["total_assets"]
        .transform(lambda s: calc_trailing(s, period)) / period
    )

    # ROE & ROA
    df["roe"] = safe_divide(
        df["profit"],
        df["avg_equity"],
    )

    df["roa"] = safe_divide(
        df["profit"],
        df["avg_assets"],
    )
    return df

def calc_bvps(
    df: pd.DataFrame,
    share_col: str = "issue_shares",
) -> pd.DataFrame:

    df = df.copy()
    df["bvps"] = df["equity"] / df[share_col]
    return df

@asset(
    io_manager_key="minio_io_manager",
    ins={
        "overview": AssetIn(["silver", "company_info", "silver_overview"]),
        "reports": AssetIn(["silver", "silver_reports"]),
        "events": AssetIn(["silver", "company_info", "silver_events"]),
    },
    group_name="gold",
    key_prefix=["gold"],
)
def gold_ticker_metric(
    overview: pd.DataFrame,
    reports: pd.DataFrame,
    events: pd.DataFrame,
) -> Output[pd.DataFrame]:

    # ======================
    # 1. EPS (TTM 4 quý)
    # ======================
    profit_df = reports[
        (reports["criteria"] == "profit") &
        (reports["year"] >= 2020)
    ][["ticker", "year", "quarter", "value"]].copy()

    profit_df = profit_df.sort_values(
        ["ticker", "year", "quarter"],
    )

    profit_df["profit_ttm"] = (
        profit_df
        .groupby("ticker")["value"]
        .transform(lambda s: calc_trailing(s, 4))
    )

    # ======================
    # 2. Adjust issued shares
    # ======================
    events = events.copy()

    events["issued_shares"] = events.apply(
        lambda r: extract_issued_shares(r["event_title"])
        if r["event_type"] == "Niêm yết thêm"
        else None,
        axis=1,
    )

    events[["event_year", "event_quarter"]] = (
        events["issue_date"].apply(lambda x: pd.Series(date_to_quarter(x, 0)))
    )

    base_share = overview.set_index("ticker")["issue_share"]
    events["issue_date"] = pd.to_datetime(events["issue_date"])
    date_fetched = pd.to_datetime(overview["date_fetched"]).max()
    events = events[events["issue_date"] < date_fetched]

    def calc_issue_share(ticker, year, quarter):
        share = base_share.loc[ticker]

        ev = events[
            (events["ticker"] == ticker) &
            (
                (events["event_year"] > year) |
                (
                    (events["event_year"] == year) &
                    (events["event_quarter"] > quarter)
                )
            )
        ]

        for _, r in ev.iterrows():
            if pd.notna(r["issued_shares"]):
                share -= r["issued_shares"]

        return share

    profit_df["issue_share_adj"] = profit_df.apply(
        lambda r: calc_issue_share(
            r["ticker"], r["year"], r["quarter"]
        ),
        axis=1,
    )

    profit_df["eps"] = (
        safe_divide(
            profit_df["profit_ttm"],
            profit_df["issue_share_adj"],
        ) * 1_000_000_000
    )

    # ======================
    # 3. ROE / ROA (trailing = 2)
    # ======================
    wide_fin = build_wide_financials(reports)

    ratio_df = calc_roe_roa_with_trailing(
        wide_fin,
        period=2,
    )

    # trailing equity 4 quý cho NIM
    ratio_df["avg_earning_assets_4q"] = (
        ratio_df
        .groupby("ticker")["earning_assets"]
        .transform(lambda s: calc_trailing(s, 4)) / 4
    )

    ratio_df = ratio_df.merge(
        overview[["ticker", "industry"]],
        on="ticker",
        how="left",
    )

    BANK_MASK = (
        ratio_df["industry"]
        .str.lower()
        .str.contains("ngân hàng", na=False)
    )

    ratio_df["nim"] = np.nan

    ratio_df.loc[BANK_MASK, "nim"] = safe_divide(
        ratio_df.loc[BANK_MASK, "interest_income"]
        + ratio_df.loc[BANK_MASK, "interest_expenses"],
        ratio_df.loc[BANK_MASK, "avg_earning_assets_4q"],
    )*4

    # ======================
    # 4. BVPS
    # ======================
    ratio_df = ratio_df.merge(
        profit_df[["ticker", "year", "quarter", "issue_share_adj"]],
        on=["ticker", "year", "quarter"],
        how="left",
    )

    ratio_df["bvps"] = (
        safe_divide(
            ratio_df["equity"],
            ratio_df["issue_share_adj"],
        ) * 1_000_000_000
    )
    # ROS
    ratio_df["ros"] = safe_divide(
        ratio_df["profit"],
        ratio_df["revenue"],
    )

    ratio_df.loc[BANK_MASK, "ros"] = np.nan

    # ======================
    # 5. Merge output dạng WIDE
    # ======================

    eps_df = profit_df[
        ["ticker", "year", "quarter", "eps"]
    ]

    ratio_out = ratio_df[
        ["ticker", "year", "quarter", "bvps", "roe", "roa", "ros", "nim"]
    ]

    final_df = eps_df.merge(
        ratio_out,
        on=["ticker", "year", "quarter"],
        how="left",
    )

    ratio_df = ratio_df.dropna(subset=["industry"])

    industry_agg = (
        ratio_df
        .groupby(["industry", "year", "quarter"])
        .agg(
            equity_sum=("equity", "sum"),
            shares_sum=("issue_share_adj", "sum"),
            profit_sum=("profit", "sum"),
            avg_equity_sum=("avg_equity", "sum"),
            avg_assets_sum=("avg_assets", "sum"),
            revenue_sum=("revenue", "sum"),
        )
        .reset_index()
    )

    industry_agg["bvps_industry"] = safe_divide(
        industry_agg["equity_sum"],
        industry_agg["shares_sum"],
    ) * 1_000_000_000

    industry_agg["roe_industry"] = safe_divide(
        industry_agg["profit_sum"],
        industry_agg["avg_equity_sum"],
    )

    industry_agg["roa_industry"] = safe_divide(
        industry_agg["profit_sum"],
        industry_agg["avg_assets_sum"],
    )

    industry_agg["ros_industry"] = safe_divide(
        industry_agg["profit_sum"],
        industry_agg["revenue_sum"],
    )
    bank_industry_mask = (
        industry_agg["industry"]
        .str.lower()
        .str.contains("ngân hàng", na=False)
    )
    industry_agg.loc[bank_industry_mask, "ros_industry"] = np.nan

    final_df = final_df.merge(
        overview[["ticker", "industry"]],
        on="ticker",
        how="left",
    )

    nim_industry_df = (
        ratio_df[BANK_MASK]
        .groupby(["industry", "year", "quarter"])
        .agg(
            interest_income_sum=("interest_income", "sum"),
            interest_expenses_sum=("interest_expenses", "sum"),
            avg_equity_4q_sum=("avg_earning_assets_4q", "sum"),
        )
        .reset_index()
    )

    nim_industry_df["nim_industry"] = safe_divide(
        nim_industry_df["interest_income_sum"]
        + nim_industry_df["interest_expenses_sum"],
        nim_industry_df["avg_equity_4q_sum"],
    )*4

    industry_agg = industry_agg.merge(
        nim_industry_df[
            ["industry", "year", "quarter", "nim_industry"]
        ],
        on=["industry", "year", "quarter"],
        how="left",
    )

    final_df = final_df.merge(
        industry_agg[
            [
                "industry", "year", "quarter",
                "bvps_industry", "roe_industry",
                "roa_industry", "ros_industry",
                "nim_industry"
            ]
        ],
        on=["industry", "year", "quarter"],
        how="left",
    )

    # ======================
    # 6. ROUNDING METRICS
    # ======================

    FOUR_DEC_COLS = [
        "roe", "roa", "ros", "nim",
        "roe_industry", "roa_industry", "ros_industry", "nim_industry",
    ]

    TWO_DEC_COLS = [
        c for c in final_df.select_dtypes("number").columns
        if c not in FOUR_DEC_COLS
    ]

    final_df[FOUR_DEC_COLS] = final_df[FOUR_DEC_COLS].round(4)
    final_df[TWO_DEC_COLS] = final_df[TWO_DEC_COLS].round(2)

    out_df = final_df[final_df["year"] >= 2021]

    return Output(
        out_df,
        metadata={"num_records": len(out_df)},
    )

@asset(
    ins={
        "gold_ticker_metric": AssetIn(
            key_prefix=["gold"]
        )
    },
    io_manager_key="psql_io_manager",
    key_prefix=["warehouse"],
    compute_kind="python",
    group_name="warehouse",
)
def warehouse_ticker_metric (gold_ticker_metric: pd.DataFrame,
) -> Output[pd.DataFrame]:

    return Output(
        gold_ticker_metric,
        metadata={
            "table": "warehosue.warehouse_ticker_metric",
            "rows_loaded": len(gold_ticker_metric),
            "unique_key": ["ticker", "year", "quarter"],
        },
    )