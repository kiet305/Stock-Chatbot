from datetime import datetime, timedelta, timezone
import pandas as pd
from dagster import asset, Output, DailyPartitionsDefinition, MetadataValue
from etl_pipeline.ops.crawling.vietstock import crawl_vietstock_news

VN_TZ = timezone(timedelta(hours=7))
daily = DailyPartitionsDefinition(
    start_date="2025-12-01",
    timezone="Asia/Ho_Chi_Minh",
    end_offset=1,
)

@asset(
    partitions_def=daily,
    io_manager_key="minio_io_manager",
    key_prefix=["bronze"],
    compute_kind="python",
    group_name="bronze",
)
def bronze_vietstock_news(context) -> Output[pd.DataFrame]:
    """
    Crawl Vietcap AI News (BRONZE)
    - Partition theo ngày (YYYY-MM-DD)
    - Crawl window: [partition_day - 1d, min(partition_day + 1d, now))
    - Dữ liệu raw, chưa dedup, chưa clean
    """
    import inspect
    context.log.info(inspect.signature(crawl_vietstock_news)) 

    # -------- Partition time window --------
    day_str = context.partition_key  # YYYY-MM-DD

    partition_day = datetime.strptime(
        day_str, "%Y-%m-%d"
    ).replace(tzinfo=VN_TZ)

    # buffer 1 ngày để tránh sót bài đăng muộn
    start_dt = partition_day - timedelta(days=1)

    # không crawl quá hiện tại khi backfill
    now = datetime.now(VN_TZ)
    end_dt = now

    context.log.info(
        f"[BRONZE] Crawl Vietstock News | "
        f"partition={day_str} | "
        f"start={start_dt.isoformat()} | "
        f"end={end_dt.isoformat()}"
    )

    # -------- Crawl --------
    records = crawl_vietstock_news(
        start_date=start_dt,
        end_date=end_dt,
    )

    context.log.info(
        f"[BRONZE] Finished | records={len(records)}"
    )

    return Output(
        records,
        metadata={
            "partition": day_str,
            "num_records": len(records),
        },
    )