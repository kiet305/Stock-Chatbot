from vnstock import Listing
import pandas as pd
import numpy as np

def get_stock_list():
    listing = Listing(source='VCI')

    df = pd.DataFrame(listing.symbols_by_exchange())
    df = df[df["type"].str.contains("STOCK", na=False)].reset_index(drop=True)

    df = df.rename(columns={
        "symbol": "ticker",
        "organ_short_name": "name",
        "exchange": "trading_floor"
    })

    df = df[['ticker', 'name', 'trading_floor']]

    # nhóm theo vốn hóa
    vn30 = listing.symbols_by_group('VN30')
    vn100 = listing.symbols_by_group('VN100')
    hnx30 = listing.symbols_by_group('HNX30')
    df["is_vn30"] = df["ticker"].isin(vn30)
    df["is_vn100"] = df["ticker"].isin(vn100)
    df["is_hnx30"] = df["ticker"].isin(hnx30)

    # nhóm theo ngành
    df_icb = listing.symbols_by_industries()
    df = df.merge(
        df_icb[["symbol", "icb_name2", "icb_name3"]],
        left_on="ticker",
        right_on="symbol",
        how="left"
    )
    df = df.drop(columns={'symbol'})
    df = df.rename(columns={'icb_name3': 'subindustry', 'icb_name2': 'industry'})
    
    return df


def normalize_overview(df: pd.DataFrame) -> pd.DataFrame:
    df_master = get_stock_list()
    df_silver = pd.merge(
        df,
        df_master,
        left_on="ticker",
        right_on="ticker",
        how="inner",
    )

    # ---------- Chuẩn hoá cap_group ----------
    df_silver["cap_group"] = np.nan

    df_silver.loc[df_silver["is_vn30"] == True, "cap_group"] = "VN30"
    df_silver.loc[
        (df_silver["is_vn30"] != True) & (df_silver["is_vn100"] == True),
        "cap_group",
    ] = "VN100"
    df_silver.loc[
        (df_silver["is_vn30"] != True)
        & (df_silver["is_vn100"] != True)
        & (df_silver["is_hnx30"] == True),
        "cap_group",
    ] = "HNX30"

    # ---------- Select final columns ----------
    df_silver = df_silver[
        [
            "ticker",
            "name",
            "trading_floor",
            "industry",
            "subindustry",
            "history",
            "company_profile",
            "issue_share",
            "cap_group",
            "date_fetched"
        ]
    ]
    return df_silver

def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    df_silver = df[['ticker', 'event_title', 'event_list_name', 'ratio', 'value',
                           'public_date', 'issue_date', 'record_date', 'exright_date']]
    df_silver = df_silver.rename(columns = {
        'event_list_name': 'event_type'
    })
    return df_silver

def normalize_shareholders(df: pd.DataFrame) -> pd.DataFrame:
    df['quantity'] = (df['quantity'] / 1_000_000).round(3)
    df_silver = df[['ticker', 'share_holder', 'quantity', 'share_own_percent', 'update_date']]
    return df_silver

def normalize_officers(df: pd.DataFrame) -> pd.DataFrame:
    df['quantity'] = (df['quantity'] / 1_000_000).round(3)
    df_silver = df[['ticker', 'officer_name', 'officer_position', 'quantity', 'officer_own_percent', 'update_date']]
    return df_silver

def normalize_info(df: pd.DataFrame, info_type: str = "overview") -> pd.DataFrame:
    func_name = f"normalize_{info_type}"
    try:
        func = globals()[func_name]
    except KeyError:
        raise ValueError(f"Unsupported info_type: {info_type}")
    return func(df)