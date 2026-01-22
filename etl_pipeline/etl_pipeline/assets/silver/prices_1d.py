import pandas as pd
from dagster import asset, AssetIn, Nothing, StaticPartitionsDefinition, Output, DailyPartitionsDefinition
from datetime import timezone, timedelta
from etl_pipeline.ops.normalize.reports import normalize_reports, convert_fact_table
from datetime import date

daily = DailyPartitionsDefinition(
    start_date="2020-01-01",
    timezone="Asia/Ho_Chi_Minh",
    end_offset=1,
)
@asset(
    partitions_def=daily,
    io_manager_key="minio_io_manager",
    ins={
        "prices": AssetIn(["bronze", "prices", "bronze_prices_1d"]),
    },
    group_name="silver",
    key_prefix=["silver"],
)
def silver_prices_1d(context, prices) -> Output[pd.DataFrame]:
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices[['ticker', 'date', 'high', 'low', 'open', 'close', 'volume']]
    return Output(
        prices,
        metadata={"num_records": len(prices)},
    )