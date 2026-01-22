import pandas as pd
import numpy as np
from dagster import asset, AssetIn, DailyPartitionsDefinition, Output, MetadataValue
from datetime import datetime

daily = DailyPartitionsDefinition(
    start_date="2020-01-01",
    timezone="Asia/Ho_Chi_Minh",
    end_offset=1,
)

INDEX_TICKERS = {
    "VNINDEX", "VN30", "VN100", "HNX30", "UPCOMINDEX"
}

def load_nearest_partition(
    io,
    asset_key,
    target_date: str,
    max_lookback: int = 10,
):
    """
    Try to load partition at target_date.
    If not exists, fallback to nearest previous date (up to max_lookback days).
    """
    dt = pd.to_datetime(target_date)

    for i in range(max_lookback + 1):
        date_str = (dt - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            return io.load_partition(asset_key, date_str), date_str
        except FileNotFoundError:
            continue

    return None, None

def calc_pct_change(
    df_today: pd.DataFrame,
    df_past: pd.DataFrame,
) -> pd.Series:
    return (df_today["close"] / df_past["close"] - 1) * 100

OFFSETS = {
    "chg_1d": 1,
    "chg_1w": 7,
    "chg_1m": 30,
    "chg_3m": 90,
    "chg_6m": 180,
    "chg_1y": 365,
    "chg_3y": 1095,
}

@asset(
    partitions_def=daily,
    io_manager_key="minio_io_manager",
    ins={
        "prices": AssetIn(["silver", "silver_prices_1d"]),
        "ticker_metric": AssetIn(["gold", "gold_ticker_metric"]),
        "overview": AssetIn(["silver", "company_info", "silver_overview"]),
    },
    group_name="gold",
    key_prefix=["gold"],
)
def gold_prices_1d(
    context,
    prices: pd.DataFrame,
    ticker_metric: pd.DataFrame,
    overview: pd.DataFrame,
) -> Output[pd.DataFrame]:

    # =========================================================
    # 0. Base
    # =========================================================
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])

    # =========================================================
    # 2. Merge ticker_metric (EPS, BVPS – prev quarter)
    # =========================================================
    metric_df = ticker_metric[["ticker", "year", "quarter", "eps", "bvps"]].copy()

    metric_df["year"] = metric_df["year"].astype(int)
    metric_df["quarter"] = metric_df["quarter"].astype(int)

    # year/quarter từ date
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # ===== helper: lùi n quý =====
    def shift_back_quarter(year, quarter, n=1):
        q = quarter - n
        y = year.copy()
        while True:
            mask = q <= 0
            if not mask.any():
                break
            q = q.where(~mask, q + 4)
            y = y.where(~mask, y - 1)
        return y, q

    # Q-1 và Q-2
    df["prev1_year"], df["prev1_quarter"] = shift_back_quarter(df["year"], df["quarter"], n=1)
    df["prev2_year"], df["prev2_quarter"] = shift_back_quarter(df["year"], df["quarter"], n=2)

    # ===== merge Q-1 =====
    m1 = metric_df.rename(columns={"year": "prev1_year", "quarter": "prev1_quarter"})
    df = df.merge(
        m1[["ticker", "prev1_year", "prev1_quarter", "eps", "bvps"]],
        on=["ticker", "prev1_year", "prev1_quarter"],
        how="left",
    ).rename(columns={"eps": "eps_prev1", "bvps": "bvps_prev1"})

    # ===== merge Q-2 =====
    m2 = metric_df.rename(columns={"year": "prev2_year", "quarter": "prev2_quarter"})
    df = df.merge(
        m2[["ticker", "prev2_year", "prev2_quarter", "eps", "bvps"]],
        on=["ticker", "prev2_year", "prev2_quarter"],
        how="left",
    ).rename(columns={"eps": "eps_prev2", "bvps": "bvps_prev2"})

    # ===== chọn eps/bvps: Q-1 -> nếu null thì Q-2 =====
    df["eps"] = df["eps_prev1"].combine_first(df["eps_prev2"])
    df["bvps"] = df["bvps_prev1"].combine_first(df["bvps_prev2"])

    # cleanup
    df = df.drop(columns=[
        "year", "quarter",
        "eps_prev1", "bvps_prev1",
        "eps_prev2", "bvps_prev2",
        "prev1_year", "prev1_quarter",
        "prev2_year", "prev2_quarter",
    ])

    # =========================================================
    # 3. Merge overview (classification + issue_share)
    # =========================================================
    df = df.merge(
        overview[["ticker", "trading_floor", "cap_group", "issue_share"]],
        on="ticker",
        how="left",
    )

    # =========================================================
    # 4. STOCK METRICS (chỉ cho cổ phiếu)
    # =========================================================
    is_stock = ~df["ticker"].isin(INDEX_TICKERS)

    df.loc[is_stock, "market_cap"] = (
        df.loc[is_stock, "close"] * df.loc[is_stock, "issue_share"] / 1_000_000
    )
    df.loc[is_stock, "pe"] = df.loc[is_stock, "close"] / df.loc[is_stock, "eps"] * 1000
    df.loc[is_stock, "pb"] = df.loc[is_stock, "close"] / df.loc[is_stock, "bvps"] * 1000

    df.loc[df["eps"] == 0, "pe"] = None
    df.loc[df["bvps"] == 0, "pb"] = None

    # =========================================================
    # 5. INDEX PE / PB – aggregate từ TOÀN THỊ TRƯỜNG
    # =========================================================
    stock_df = df[is_stock].copy()

    stock_df["market_cap_raw"] = stock_df["close"] * stock_df["issue_share"]
    stock_df["earnings"] = stock_df["eps"] * stock_df["issue_share"]
    stock_df["equity"] = stock_df["bvps"] * stock_df["issue_share"]

    def calc_index_pe_pb(x: pd.DataFrame):
        mcap = x["market_cap_raw"].sum()
        earnings = x["earnings"].sum()
        equity = x["equity"].sum()

        pe = mcap / earnings if earnings and earnings > 0 else None
        pb = mcap / equity if equity and equity > 0 else None

        pe = pe * 1000 if pe is not None else None
        pb = pb * 1000 if pb is not None else None

        return pe, pb

    index_rows = []

    # VN30
    vn30 = stock_df[stock_df["cap_group"] == "VN30"]
    index_rows.append(("VN30", *calc_index_pe_pb(vn30)))

    # VN100 (VN30 ⊂ VN100)
    vn100 = stock_df[stock_df["cap_group"].isin(["VN30", "VN100"])]
    index_rows.append(("VN100", *calc_index_pe_pb(vn100)))

    # HNX30
    hnx30 = stock_df[stock_df["cap_group"] == "HNX30"]
    index_rows.append(("HNX30", *calc_index_pe_pb(hnx30)))

    # VNINDEX (toàn thị trường)
    vnindex = stock_df[
        stock_df["trading_floor"].isin(["HOSE", "HNX", "UPCOM"])
    ]
    index_rows.append(("VNINDEX", *calc_index_pe_pb(vnindex)))

    # UPCOMINDEX
    upcom = stock_df[stock_df["trading_floor"] == "UPCOM"]
    index_rows.append(("UPCOMINDEX", *calc_index_pe_pb(upcom)))

    index_metric_df = pd.DataFrame(
        index_rows,
        columns=["ticker", "pe_index", "pb_index"],
    )

    df = df.merge(index_metric_df, on="ticker", how="left")
    is_index = df["ticker"].isin(INDEX_TICKERS)

    # Với index → dùng pe_index / pb_index
    df.loc[is_index, "pe"] = df.loc[is_index, "pe_index"]
    df.loc[is_index, "pb"] = df.loc[is_index, "pb_index"]
    # =========================================================
    # 6. Cleanup + rounding
    # =========================================================
    df = df.drop(
        columns=[
            "eps", "bvps", "issue_share",
            'pe_index', 'pb_index',
            'cap_group',
        ]
    )
    # =========================================================
    # 7. PRICE CHANGE (%), WITH FALLBACK PARTITION
    # =========================================================
    io = context.resources.minio_io_manager
    asset_key = context.asset_key
    today = context.partition_key  # 'YYYY-MM-DD'
    df_today = df[["ticker", "close"]].set_index("ticker")

    used_partitions = {}

    for col, days in OFFSETS.items():
        target_date = (
            pd.to_datetime(today) - pd.Timedelta(days=days)
        ).strftime("%Y-%m-%d")

        df_past, used_date = load_nearest_partition(
            io=io,
            asset_key=asset_key,
            target_date=target_date,
            max_lookback=10,
        )

        used_partitions[col] = used_date

        if df_past is None:
            df_today[col] = None
            continue

        df_past = df_past[["ticker", "close"]].set_index("ticker")
        df_today[col] = calc_pct_change(df_today, df_past)


    # merge back
    df = df.merge(
        df_today.reset_index()[["ticker"] + list(OFFSETS.keys())],
        on="ticker",
        how="left",
    )

    EXCLUDE_COLS = {"volume"}
    num_cols = df.select_dtypes(include=[np.number]).columns
    round_cols = [c for c in num_cols if c not in EXCLUDE_COLS]
    df[round_cols] = df[round_cols].round(2)

    return Output(
        df,
        metadata={
            "num_records": len(df),
            "price_change_partitions": MetadataValue.json(used_partitions),
        },
    )


@asset(
    partitions_def=daily,
    ins={
        "gold_prices_1d": AssetIn(
            key_prefix=["gold"]
        )
    },
    io_manager_key="psql_io_manager",
    key_prefix=["warehouse"],
    compute_kind="python",
    group_name="warehouse",
)
def warehouse_prices_1d (gold_prices_1d: pd.DataFrame,
) -> Output[pd.DataFrame]:

    return Output(
        gold_prices_1d,
        metadata={
            "table": "warehouse.warehouse_prices_1d",
            "rows_loaded": len(gold_prices_1d),
            "unique_key": ["ticker", "date"]
        },
    )