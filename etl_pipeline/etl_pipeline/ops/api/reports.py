import pandas as pd
import time
from requests.exceptions import ConnectionError
from datetime import datetime
from vnstock import Finance, Listing

def get_stock_list():
    listing = Listing(source="VCI")
    df = pd.DataFrame(listing.symbols_by_exchange())
    df = df[df["type"] == "STOCK"]
    return df["symbol"].tolist()

def retry_call(fn, logger, max_retries=10, rate_limit_sleep=20):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()

        except SystemExit:
            if attempt < max_retries:
                logger.warning(
                    f"Rate limit hit. Retry {attempt}/{max_retries} "
                    f"after {rate_limit_sleep}s"
                )
                time.sleep(rate_limit_sleep)
            else:
                logger.error("Rate limit hit. Max retries reached.")
                return None

        except ConnectionError:
            logger.warning("Connection error (502/504). Skip symbol")
            return None

        except Exception as e:
            logger.error(f"Unexpected error. Skip symbol: {e}")
            return None
        
def get_report(
    *,
    context,
    tickers: list[str],
    report_type: str = "is",
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Bronze layer financial reports
    """

    REPORT_TYPE = {
        "is": "INCOME STATEMENT",
        "bs": "BALANCE SHEET",
        "cf": "CASH FLOW",
    }

    bronze_frames = []
    fetched_symbols = []
    skipped_symbols = []

    total = len(tickers)
    report_name = REPORT_TYPE.get(report_type, report_type.upper())

    context.log.info(
        f"Start crawling {report_name} | symbols={total}"
    )

    for idx, ticker in enumerate(tickers, start=1):
        context.log.info(
            f"[{idx}/{total}] Fetching {report_name} for {ticker}"
        )

        # --- chọn đúng API call ---
        if report_type == "is":
            fn = lambda t=ticker: Finance(
                symbol=t, source="VCI"
            ).income_statement(period="quarter", lang="vi")

        elif report_type == "bs":
            fn = lambda t=ticker: Finance(
                symbol=t, source="VCI"
            ).balance_sheet(period="quarter", lang="vi")

        elif report_type == "cf":
            fn = lambda t=ticker: Finance(
                symbol=t, source="VCI"
            ).cash_flow(period="quarter", lang="vi")

        else:
            raise ValueError(
                "report_type must be one of: 'is', 'bs', 'cf'"
            )

        # --- gọi API với retry ---
        df_raw = retry_call(fn, logger=context.log)

        if df_raw is None or df_raw.empty:
            context.log.warning(f"Skip {ticker}")
            skipped_symbols.append(ticker)
            continue

        context.log.info(
            f"Fetched {report_type.upper()} for {ticker} "
            f"({len(df_raw)} rows)"
        )

        fetched_symbols.append(ticker)

        # --- Bronze metadata ---
        df_raw = df_raw.copy()
        df_raw["ticker"] = ticker
        df_raw["report_type"] = report_type
        df_raw["date_fetched"] = datetime.now()

        bronze_frames.append(df_raw)

        # --- optional limit ---
        if limit and len(fetched_symbols) >= limit:
            context.log.warning(f"Reached limit = {limit}")
            break

    # --- summary ---
    context.log.info("INGEST SUMMARY")
    context.log.info(f"Success ({len(fetched_symbols)}): {fetched_symbols}")
    context.log.info(f"Skipped ({len(skipped_symbols)}): {skipped_symbols}")

    if not bronze_frames:
        context.log.warning("No data fetched")
        return pd.DataFrame()

    return pd.concat(bronze_frames, ignore_index=True)