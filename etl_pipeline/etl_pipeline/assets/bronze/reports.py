import pandas as pd
from etl_pipeline.ops.api.reports import get_report, get_stock_list
from dagster import asset, StaticPartitionsDefinition, Output
from datetime import date

def get_quarters(
        current: str | None = None,
        n = 21
):
    # Lấy quý trước
    if current is None:
        year = date.today().year
        month = date.today().month
        prev_quarter = (month - 1) // 3
    else:
        year, prev_quarter = map(int, current.split("-Q"))

    quarters = []

    # Lặp theo số quý
    for _ in range(n):
        if prev_quarter == 0:
            prev_quarter = 4
            year -= 1    
        quarters.append(f"{year}-Q{prev_quarter}")
        prev_quarter -= 1

    return quarters

quarters = get_quarters()
report_partitions = StaticPartitionsDefinition(quarters)

def detect_inactive_symbols(
    *,
    io,
    asset_key,
    all_tickers: set[str],
    lookback=4,
) -> set[str]:
    existing_partitions = sorted(io.list_partitions(asset_key))
    if len(existing_partitions) < lookback:
        # chưa đủ dữ liệu → coi như tất cả active
        return set()

    recent_partitions = existing_partitions[-lookback:]

    active = set()
    for p in recent_partitions:
        df = io.load_partition(asset_key, p)
        # union các mã có ít nhất 1 bctc trong 4 quý gần nhất  
        active |= set(df["CP"].unique())

    inactive = all_tickers - active
    return inactive

REPORT_NAME_MAP = {
    "is": "income_statement",
    "bs": "balance_sheet",
    "cf": "cash_flow",
}

def bronze_reports(report_type: str):
    asset_suffix = REPORT_NAME_MAP[report_type]
    asset_name = f"bronze_{asset_suffix}"

    @asset(
        name=asset_name,
        key_prefix=["bronze", "reports"],
        partitions_def=report_partitions,
        io_manager_key="minio_io_manager",
        group_name="bronze",
    )
    def _asset(context):
        io = context.resources.minio_io_manager
        asset_key = context.asset_key

        existing_partitions = io.list_partitions(asset_key)
        year, quarter = map(int, context.partition_key.split("-Q"))

        all_tickers = set(get_stock_list())

        # Full load mode (lần chạy đầu tiên)
        if not existing_partitions:
            context.log.info("FULL LOAD MODE")

            df = get_report(
                context=context,
                report_type=report_type,
                tickers=sorted(all_tickers),
            )

            # ghi toàn bộ lịch sử thành parquet partitions
            for (y, q), df_q in df.groupby(["Năm", "Kỳ"]):
                p = f"{y}-Q{q}"
                if p in quarters:
                    io.write_partition(asset_key, p, df_q)

            # partition hiện tại không cần ghi thêm
            context.log.info("Full load done, skip partition materialization")
            context.log.info(
                "Full load done, skip partition materialization\n"
                f"Tickers={len(all_tickers)}\n"
                f"Rows={len(df)}\n"
                f"Partitions_written="
                f"{df[['Năm', 'Kỳ']].drop_duplicates().shape[0]}"
            )
            return


        # Incremental mode
        context.log.info("INCREMENTAL MODE")

        inactive = detect_inactive_symbols(
            io=io,
            asset_key=asset_key,
            all_tickers=all_tickers,
            lookback=4,
        )

        context.log.info(f"Tickers inactive: {len(inactive)}")

        try:
            df_existing = io.load_partition(asset_key, context.partition_key)
            existing_symbols = set(df_existing["CP"].unique())
        except FileNotFoundError:
            df_existing = pd.DataFrame()
            existing_symbols = set()

        tickers = sorted(all_tickers - inactive - existing_symbols)
        context.log.info(f"Tickers to be crawled: {len(tickers)}")

        if not tickers:
            context.log.info("No new symbols to crawl")
            return

        df = get_report(
            context=context,
            report_type=report_type,
            tickers=tickers,
        )

        df_part = df[
            (df["Năm"] == year) &
            (df["Kỳ"] == quarter)
        ]

        df_merged = pd.concat(
            [
                df_existing[~df_existing["CP"].isin(df_part["CP"])],
                df_part,
            ],
            ignore_index=True,
        )

        return Output(
            df_merged,
            metadata={
                "mode": "incremental",
                "year": year,
                "quarter": quarter,
                "tickers_crawled": len(tickers),
                "rows": len(df_part),
                "api_called": True,
            },
        )

    return _asset