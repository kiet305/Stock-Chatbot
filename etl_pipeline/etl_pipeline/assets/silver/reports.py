import pandas as pd
from dagster import asset, AssetIn, Nothing, StaticPartitionsDefinition, Output
from datetime import timezone, timedelta
from etl_pipeline.ops.normalize.reports import normalize_reports, convert_fact_table
from datetime import date

@asset(
    io_manager_key="minio_io_manager",
    ins={
        "bs": AssetIn(["bronze", "reports", "bronze_balance_sheet"],
                      metadata={"load_all_partitions": True}),
        "is_": AssetIn(["bronze", "reports", "bronze_income_statement"],
                      metadata={"load_all_partitions": True}),
        "cf": AssetIn(["bronze", "reports", "bronze_cash_flow"],
                      metadata={"load_all_partitions": True}),
    },
    group_name="silver",
    key_prefix=["silver"],
)
def silver_reports(context, bs, is_, cf) -> Output[pd.DataFrame]:
    bs_bank, bs_non_bank = normalize_reports(bs, 'bs')
    is_bank, is_non_bank = normalize_reports(is_, 'is')
    cf_bank, cf_non_bank = normalize_reports(cf, 'cf')

    df = pd.concat(
        [
            convert_fact_table(bs_bank),
            convert_fact_table(bs_non_bank),
            convert_fact_table(is_bank),
            convert_fact_table(is_non_bank),
            convert_fact_table(cf_bank),
            convert_fact_table(cf_non_bank),
        ],
        ignore_index=True,
    )

    return Output(
        df,
        metadata={"num_records": len(df)},
    )
