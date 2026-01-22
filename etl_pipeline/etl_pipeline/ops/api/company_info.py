import pandas as pd
import time
from requests.exceptions import ConnectionError
from datetime import datetime
from vnstock import Listing, Company

def retry_call(
    fn,
    logger,
    max_retries=3,
    retry_sleep=5,
    rate_limit_sleep=20,
):
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

        except ConnectionError as e:
            if attempt < max_retries:
                logger.warning(
                    f"Connection error. Retry {attempt}/{max_retries} "
                    f"after {retry_sleep}s"
                )
                time.sleep(retry_sleep)
            else:
                logger.error(
                    "Connection error. Max retries reached. Skip symbol"
                )
                return None

        except Exception as e:
            logger.error(f"Unexpected error. Skip symbol: {e}")
            return None

def get_stock_list():
    listing = Listing(source="VCI")
    df = pd.DataFrame(listing.symbols_by_exchange())
    df = df[df["type"] == "STOCK"]
    return df["symbol"].tolist()

def get_company_information(
    context,
    tickers: list[str],
    type: str = "overview",
    limit: int | None = None,
):
    bronze_frames = []
    fetched_symbols = []
    skipped_symbols = []
    total = len(tickers)

    TYPE_KWARGS = {
        "officers": {"filter_by": "working"},
    }

    context.log.info(
        f"Start crawling {type} | symbols={total}"
    )

    for idx, ticker in enumerate(tickers, start=1):
        context.log.info(
            f"[{idx}/{total}] Fetching {type.upper()} for {ticker}"
        )

        def fn():
            company = Company(symbol=ticker, source="VCI")
            if not hasattr(company, type):
                raise ValueError(f"Unsupported company API: {type}")

            method = getattr(company, type)
            kwargs = TYPE_KWARGS.get(type, {})
            return method(**kwargs)

        df = retry_call(fn, logger=context.log)

        if df is None or df.empty:
            context.log.warning(f"Skip {ticker}")
            skipped_symbols.append(ticker)
            continue

        context.log.info(f"Fetched {type} for: {ticker}")
        fetched_symbols.append(ticker)

        df = df.copy()
        df["ticker"] = ticker
        df["date_fetched"] = datetime.now()

        bronze_frames.append(df)

        if limit and len(fetched_symbols) >= limit:
            context.log.warning(f"Reached limit = {limit}")
            break

    context.log.info("INGEST SUMMARY")
    context.log.info(f"Success ({len(fetched_symbols)}): {fetched_symbols}")
    context.log.info(f"Skipped ({len(skipped_symbols)}): {skipped_symbols}")

    if not bronze_frames:
        context.log.warning("No data fetched")
        return pd.DataFrame()

    return pd.concat(bronze_frames, ignore_index=True)