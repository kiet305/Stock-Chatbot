from datetime import datetime, timedelta, timezone
import pandas as pd
from dagster import asset, Output, AssetIn
from etl_pipeline.ops.api.company_info import get_company_information, get_stock_list
from etl_pipeline.ops.normalize.company_info import normalize_info

def gold_company_info(info_type: str):
    @asset(
        name = f"gold_{info_type}",
        ins = {"info": AssetIn(["silver", "company_info", f"silver_{info_type}"])},
        io_manager_key="minio_io_manager",
        key_prefix=["gold", "company_info"],
        compute_kind="python",
        group_name="gold",
    )
    def _asset(context, info):
        return Output(
            info,
            metadata={
                "info_type": info_type,
                "num_records": len(info),
            },
        )
    return _asset

def warehouse_company_info(info_type: str):
    @asset(
        name=f"warehouse_{info_type}",
        ins={
            f"gold_{info_type}": AssetIn(
                key_prefix=["gold", "company_info"]
            )
        },
        io_manager_key="psql_io_manager",
        key_prefix=["warehouse", "company_info"],
        compute_kind="python",
        group_name="warehouse",
    )
    def _asset(context, **kwargs):
        df = kwargs[f"gold_{info_type}"]

        return Output(
            df,
            metadata={
                "table": f"warehouse.warehouse_{info_type}",
                "rows_loaded": len(df),
            },
        )

    return _asset