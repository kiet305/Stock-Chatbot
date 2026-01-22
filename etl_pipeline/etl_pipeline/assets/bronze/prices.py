import pandas as pd
from etl_pipeline.ops.api.prices import get_prices, get_stock_list
from dagster import asset, DailyPartitionsDefinition, Output
from datetime import date, timedelta, timezone, datetime

VN_TZ = timezone(timedelta(hours=7))
daily = DailyPartitionsDefinition(
    start_date="2020-01-01",
    timezone="Asia/Ho_Chi_Minh",
    end_offset=1,
)

def detect_inactive_symbols(
    *,
    io,
    asset_key,
    all_tickers: set[str],
    lookback=10,
) -> set[str]:
    existing_partitions = sorted(io.list_partitions(asset_key))
    if len(existing_partitions) < lookback:
        # chưa đủ dữ liệu → coi như tất cả active
        return set()

    recent_partitions = existing_partitions[-lookback:]

    active = set()
    for p in recent_partitions:
        df = io.load_partition(asset_key, p)
        # union các mã có ít nhất 1 phiên khớp lệnh trong 10 phiên gần nhất  
        active |= set(df["ticker"].unique())

    inactive = all_tickers - active
    return inactive

INDEX_TICKERS = {
    'VNINDEX', 'VN30', 'VN100', 'HNX30', 'UPCOMINDEX'
}

def build_tickers(*, all_tickers: set[str], inactive: set[str] | None = None) -> list[str]:
    inactive = inactive or set()

    # cổ phiếu thường (active)
    stocks = sorted(
        t for t in all_tickers
        if t not in INDEX_TICKERS and t not in inactive
    )

    # chỉ số (luôn ở cuối)
    indices = sorted(
        t for t in all_tickers
        if t in INDEX_TICKERS
    )

    return stocks + indices

@asset(
    key_prefix=["bronze", "prices"],
    partitions_def=daily,
    io_manager_key="minio_io_manager",
    group_name="bronze",
)
def bronze_prices_1d(context):
    dates = context.partition_key

    io = context.resources.minio_io_manager
    asset_key = context.asset_key
    existing_partitions = io.list_partitions(asset_key)

    all_tickers = set(get_stock_list())
    all_tickers |= INDEX_TICKERS

    # Full load mode (lần chạy đầu tiên)
    if not existing_partitions:
        context.log.info("FULL LOAD MODE")

        df = get_prices(
            context=context,
            interval='1d',
            tickers=build_tickers(all_tickers=all_tickers),
            start_date = "2020-01-01",
            end_date = date.today().strftime("%Y-%m-%d"),
        )

        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"] >= pd.Timestamp("2020-01-01")]
        df = df.rename(columns={
            'time': 'date'
        })

        # ghi toàn bộ lịch sử thành parquet partitions
        for t, df_part in df.groupby("date"):
            p = t.strftime("%Y-%m-%d")
            io.write_partition(asset_key, p, df_part)

        # partition hiện tại không cần ghi thêm
        context.log.info("Full load done, skip partition materialization")
        context.log.info(
            "Full load done, skip partition materialization\n"
            f"Tickers={len(all_tickers)}\n"
            f"Rows={len(df)}\n"
            f"Partitions_written="
            f"{df[['date']].drop_duplicates().shape[0]}"
        )

        return

    # Incremental mode
    context.log.info("INCREMENTAL MODE")
    d = datetime.strptime(dates, "%Y-%m-%d").date()
    if d.weekday() >= 5:
        context.log.info(f"No data for weekends: {dates}")
        return

    inactive = detect_inactive_symbols(
        io=io,
        asset_key=asset_key,
        all_tickers=all_tickers,
        lookback=10,
    )

    context.log.info(f"Tickers inactive: {len(inactive)}")
    context.log.info(f"Partition_key: {context.partition_key}")

    try:
        df_existing = io.load_partition(asset_key, context.partition_key)
        context.log.info(f"Data updated for {dates}. Stop materializing")
        return
    except FileNotFoundError:
        df_existing = pd.DataFrame()
        tickers = build_tickers(
            all_tickers=all_tickers,
            inactive=inactive,
        )

    context.log.info(f"Tickers to be crawled: {len(tickers)}")
                     
    df = get_prices(
        context=context,
        interval='1d',
        tickers=tickers,
        start_date=dates,
        end_date=dates,
    )
    df["time"] = df["time"].dt.strftime("%Y-%m-%d")

    df_part = df[
        (df["time"] == dates)
    ]
    df_part = df_part.rename(columns={
        'time': 'date',
    })

    return Output(
        df_part,
        metadata={
            "mode": "incremental",
            "date": dates,
            "tickers_crawled": len(tickers),
            "rows": len(df_part),
            "api_called": True,
        },
    )
