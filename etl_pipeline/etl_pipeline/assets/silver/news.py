import pandas as pd
from dagster import asset, Output, AssetIn, DailyPartitionsDefinition
from datetime import timezone, timedelta
from etl_pipeline.ops.normalize.vietstock import normalize_vietstock_news
from etl_pipeline.ops.normalize.vietcap import normalize_vietcap_news

VN_TZ = timezone(timedelta(hours=7))

daily = DailyPartitionsDefinition(
    start_date="2025-12-01",
    timezone="Asia/Ho_Chi_Minh",
    end_offset=1,
)

@asset(
    partitions_def=daily,
    io_manager_key="minio_io_manager",
    ins={
        "bronze_vietstock_news": AssetIn(key_prefix=["bronze"]),
        "bronze_vietcap_news": AssetIn(key_prefix=["bronze"]),
    },
    compute_kind="Pandas",
    key_prefix=["silver"],
    group_name="silver",
)
def silver_news(
    context,
    bronze_vietstock_news: pd.DataFrame,
    bronze_vietcap_news: pd.DataFrame
):
    context.log.info("Started asset: silver_vietstock_news_tickers")

    io = context.resources.minio_io_manager
    asset_key = context.asset_key

    # 1️⃣ Normalize Bronze → Silver
    df1 = normalize_vietcap_news(bronze_vietcap_news)
    df = normalize_vietstock_news(bronze_vietstock_news)

    # 2️⃣ Chỉ giữ cột cần thiết
    df = df[
        [
            'url', 'title', 'date_posted', 'tickers', 'section', 'tags', 'summary',
        ]
    ]
    df['source'] = 'Vietstock'

    # 3️⃣ Chuẩn hoá tickers → list
    def normalize_tickers(x):
        if isinstance(x, list):
            return [t.strip().upper() for t in x if t] or [None]
        if isinstance(x, str):
            tickers = [t.strip().upper() for t in x.split(",") if t.strip()]
            return tickers or [None]
        return [None]


    df["tickers"] = df["tickers"].apply(normalize_tickers)

    # 4️⃣ EXPLODE → 1 ticker / row
    df = (
        df
        .explode("tickers")
        .rename(columns={"tickers": "ticker"})
        .reset_index(drop=True)
    )

    df = pd.concat([df1, df])

    # 5️⃣ Partition window
    partition_date = pd.to_datetime(context.partition_key).date()
    prev_date = partition_date - timedelta(days=1)

    df["date_posted"] = (
        pd.to_datetime(df["date_posted"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.tz_localize(None)
        .dt.date
    )

    df_prev = df[df["date_posted"] == prev_date]
    df_curr = df[df["date_posted"] == partition_date]

    # 6️⃣ Ghi đè CẢ 2 PARTITION
    io.write_partition(asset_key, str(prev_date), df_prev)
    io.write_partition(asset_key, str(partition_date), df_curr)

    context.log.info(
        f"Overwritten partitions: {prev_date}, {partition_date}\n"
        f"Today rows (news × tickers): {len(df_curr)}\n"
        f"Yesterday rows (news × tickers): {len(df_prev)}"
    )

    return