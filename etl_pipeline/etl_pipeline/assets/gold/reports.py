import pandas as pd
from dagster import asset, AssetIn, Nothing, DailyPartitionsDefinition, Output, MetadataValue
from datetime import timezone, timedelta
from datetime import date, datetime
import re

@asset(
    io_manager_key="minio_io_manager",
    ins={
        "reports": AssetIn(["silver", "silver_reports"]),
    },
    group_name="gold",
    key_prefix=["gold"],
)
def gold_reports(
    reports: pd.DataFrame,
) -> Output[pd.DataFrame]:
    
    return Output(
        reports,
        metadata={"num_records": len(reports)},
    )

@asset(
    ins={
        "gold_reports": AssetIn(
            key_prefix=["gold"]
        )
    },
    io_manager_key="psql_io_manager",
    key_prefix=["warehouse"],
    compute_kind="python",
    group_name="warehouse",
)
def warehouse_reports (gold_reports: pd.DataFrame,
) -> Output[pd.DataFrame]:

    return Output(
        gold_reports,
        metadata={
            "table": "warehouse.warehouse_reports",
            "rows_loaded": len(gold_reports),
            "unique_key": ["ticker", "year", "quarter", "criteria"],
        },
    )