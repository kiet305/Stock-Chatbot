import pandas as pd
from dagster import asset, AssetIn, Nothing, DailyPartitionsDefinition, Output, MetadataValue
from datetime import timezone, timedelta
from datetime import date, datetime
import re
import json
import numpy as np

daily = DailyPartitionsDefinition(
    start_date="2025-12-01",
    timezone="Asia/Ho_Chi_Minh",
    end_offset=1,
)

@asset(
    partitions_def=daily,
    io_manager_key="minio_io_manager",
    ins={
        "news": AssetIn(["silver", "silver_news"]),
    },
    group_name="gold",
    key_prefix=["gold"],
)
def gold_news(
    news: pd.DataFrame,
) -> Output[pd.DataFrame]:
    
    return Output(
        news,
        metadata={"num_records": len(news)},
    )


@asset(
    partitions_def=daily,
    ins={
        "gold_news": AssetIn(
            key_prefix=["gold"]
        )
    },
    io_manager_key="psql_io_manager",
    key_prefix=["warehouse"],
    compute_kind="python",
    group_name="warehouse",
)
def warehouse_news (gold_news: pd.DataFrame,
) -> Output[pd.DataFrame]:
    gold_news["tags"] = gold_news["tags"].apply(
        lambda x: json.dumps(x.tolist(), ensure_ascii=False)
        if isinstance(x, np.ndarray)
        else x
    )

    return Output(
        gold_news,
        metadata={
            "table": "warehouse.warehouse_news",
            "rows_loaded": len(gold_news),
            "unique_key": ["url"],
        },
    )