from datetime import datetime, timedelta, timezone
import pandas as pd
from dagster import asset, Output, AssetIn
from etl_pipeline.ops.api.company_info import get_company_information, get_stock_list
from etl_pipeline.ops.normalize.company_info import normalize_info

def silver_company_info(info_type: str):
    @asset(
        name = f"silver_{info_type}",
        ins = {"info": AssetIn(["bronze", "company_info", f"bronze_{info_type}"])},
        io_manager_key="minio_io_manager",
        key_prefix=["silver", "company_info"],
        compute_kind="python",
        group_name="silver",
    )
    def _asset(context, info):
        info_df = normalize_info(info, info_type=info_type)
        return Output(
            info_df,
            metadata={
                "info_type": info_type,
                "num_records": len(info_df),
            },
        )
    return _asset