import pandas as pd
import time
from requests.exceptions import ConnectionError
from datetime import datetime
from vnstock import Finance, Listing, Quote

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
        
def get_prices(
    *,
    context,
    tickers: list[str],
    limit: int | None = None,
    interval: str = '1d',
    start_date: str,
    end_date:  str
) -> pd.DataFrame:
    """
    Bronze layer financial reports
    """
    bronze_frames = []
    fetched_symbols = []
    skipped_symbols = []

    total = len(tickers)

    context.log.info(
        f"Start fetching {interval} prices for symbols={total}"
    )

    for idx, ticker in enumerate(tickers, start=1):
        context.log.info(
            f"[{idx}/{total}] Fetching {interval} prices for {ticker}"
        )

        # --- chọn đúng API call ---
        if interval == "1d":
            fn = lambda t=ticker: Quote(
                symbol=t, source='VCI').history(
                start=start_date, end=end_date, interval='1d')

        elif interval == "5m":
            fn = lambda t=ticker: Quote(
                symbol=t, source='VCI').history(
                start=start_date, end=end_date, interval='5m')

        else:
            raise ValueError(
                "interval must be one of: '1d' or '5m'"
            )

        # --- gọi API với retry ---
        df_raw = retry_call(fn, logger=context.log)

        if df_raw is None or df_raw.empty:
            context.log.warning(f"Skip {ticker}")
            skipped_symbols.append(ticker)
            continue

        context.log.info(
            f"Fetched {interval}  for {ticker} "
            f"({len(df_raw)} rows)"
        )

        fetched_symbols.append(ticker)

        # --- Bronze metadata ---
        df_raw = df_raw.copy()
        df_raw["ticker"] = ticker
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