# IO Managers
from .resources.minio_io_manager import MinIOIOManager
from .resources.psql_io_manager import PostgreSQLIOManager

from config.config import MINIO_CONFIG,PSQL_CONFIG

# Assets

from etl_pipeline.assets.bronze.vietcap import bronze_vietcap_news
from etl_pipeline.assets.bronze.vietstock import bronze_vietstock_news
from etl_pipeline.assets.bronze.reports import bronze_reports
from etl_pipeline.assets.bronze.company_info import bronze_company_info
from etl_pipeline.assets.bronze.prices import bronze_prices_1d
from etl_pipeline.assets.silver.news import silver_news
from etl_pipeline.assets.silver.reports import silver_reports
from etl_pipeline.assets.silver.prices_1d import silver_prices_1d
from etl_pipeline.assets.silver.company_info import silver_company_info
from etl_pipeline.assets.gold.ticker_metric import gold_ticker_metric, warehouse_ticker_metric
from etl_pipeline.assets.gold.prices_1d import gold_prices_1d, warehouse_prices_1d
from etl_pipeline.assets.gold.company_info import gold_company_info, warehouse_company_info
from etl_pipeline.assets.gold.news import gold_news, warehouse_news
# from etl_pipeline.assets.gold.vietstock import gold_vietstock_tickers, gold_vietstock_news
from etl_pipeline.assets.gold.reports import gold_reports, warehouse_reports

from dagster import Definitions, load_assets_from_modules

bronze_income_statement = bronze_reports("is")
bronze_balance_sheet   = bronze_reports("bs")
bronze_cash_flow       = bronze_reports("cf")

bronze_company_overview = bronze_company_info("overview")
bronze_company_shareholders = bronze_company_info("shareholders")
bronze_company_subsidiaries = bronze_company_info("subsidiaries")
bronze_company_events = bronze_company_info("events")
bronze_company_officers = bronze_company_info("officers")

silver_events = silver_company_info('events')
silver_overview = silver_company_info('overview')
silver_shareholders = silver_company_info('shareholders')
silver_officers = silver_company_info('officers')

gold_events = gold_company_info('events')
gold_overview = gold_company_info('overview')
gold_shareholders = gold_company_info('shareholders')
gold_officers = gold_company_info('officers')

warehouse_events = warehouse_company_info("events")
warehouse_overview = warehouse_company_info("overview")
warehouse_shareholders = warehouse_company_info("shareholders")
warehouse_officers = warehouse_company_info("officers")


defs = Definitions(
    assets=[
        warehouse_ticker_metric,
        warehouse_prices_1d,
        warehouse_reports,
        warehouse_news,
        warehouse_events,
        warehouse_overview,
        warehouse_shareholders,
        warehouse_officers,
        gold_reports,
        gold_news,
        gold_events,
        gold_overview,
        gold_shareholders,
        gold_officers,
        gold_prices_1d,
        gold_ticker_metric,
        silver_events,
        silver_shareholders,
        silver_overview,
        silver_officers,
        silver_reports,
        silver_news,
        silver_prices_1d,
        bronze_prices_1d,
        bronze_income_statement,
        bronze_balance_sheet,
        bronze_cash_flow,
        bronze_vietstock_news,
        bronze_vietcap_news,
        bronze_company_overview,
        bronze_company_shareholders,
        bronze_company_subsidiaries,
        bronze_company_events,
        bronze_company_officers,
    ],
    resources = {
        "minio_io_manager": MinIOIOManager(MINIO_CONFIG),
        "psql_io_manager": PostgreSQLIOManager(PSQL_CONFIG)
    }
)