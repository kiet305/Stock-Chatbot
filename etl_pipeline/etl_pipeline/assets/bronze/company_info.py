from datetime import datetime, timedelta, timezone
import pandas as pd
from dagster import asset, Output
from etl_pipeline.ops.api.company_info import get_company_information, get_stock_list

def bronze_company_info(info_type: str):
    @asset(
        name = f"bronze_{info_type}",
        io_manager_key="minio_io_manager",
        key_prefix=["bronze", "company_info"],
        compute_kind="python",
        group_name="bronze",
    )
    def _asset(context):
        """
        Crawl Company's Overview Information
        """
        # -------- Crawl --------
        records = get_company_information(
            context,
            tickers=get_stock_list(),
            info_type=info_type,
        )

        return Output(
            records,
            metadata={
                "info_type": info_type,
                "num_records": len(records),
            },
        )

    return _asset