import numpy as np
import pandas as pd
import re
import time
from datetime import date, datetime, timedelta, timezone

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Tuple, Optional, Dict
import logging
import math

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException, JavascriptException

import logging
logger = logging.getLogger("vietcap_bronze")
logger.setLevel(logging.INFO)

BASE = "https://trading.vietcap.com.vn"
AI_NEWS_PAGE = "https://trading.vietcap.com.vn/ai-news/market"
POST_DETAIL_KEY = "/ai-news/post-detail/"
TZ = timezone(timedelta(hours=7))

def init_browser():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,850")

    # Khởi tạo ChromeDriver với webdriver-manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(options=options, service=service)
    return driver

def parse_to_date(s: str, now: datetime | None = None) -> date | None:
    if not s:
        return None
    if now is None:
        now = datetime.now(TZ)

    s = s.strip().lower()

    # 1) Tương đối
    for pattern, unit in [
        (r"(\d+)\s*giờ\s*trước",   "hours"),
        (r"(\d+)\s*phút\s*trước",  "minutes"),
        (r"(\d+)\s*giây\s*trước",  "seconds"),
    ]:
        m = re.search(pattern, s)
        if m:
            dt = now - timedelta(**{unit: int(m.group(1))})
            return dt.astimezone(TZ).date()

    # 2) Tuyệt đối: bắt date ở bất kỳ đâu trong chuỗi (ưu tiên có giờ)
    m_dt = re.search(r"(\d{1,2}/\d{1,2}/\d{4})\s*[,\s]*([0-2]?\d:[0-5]\d)?", s)
    if m_dt:
        dpart = m_dt.group(1)
        tpart = m_dt.group(2)
        if tpart:
            try:
                return datetime.strptime(f"{dpart} {tpart}", "%d/%m/%Y %H:%M").replace(tzinfo=TZ).date()
            except ValueError:
                pass
        try:
            return datetime.strptime(dpart, "%d/%m/%Y").replace(tzinfo=TZ).date()
        except ValueError:
            pass

def _to_local_date(x: datetime | date) -> date:
    """Ép mọi kiểu (datetime/date) về date theo TZ=UTC+7 để so sánh theo ngày."""
    if isinstance(x, datetime):
        return x.astimezone(TZ).date()
    return x  # đã là date

def discover_article_links_infinite_scroll(
    start_url=AI_NEWS_PAGE,
    *,
    max_rounds: int = 200,
    step_px: int = 2000,
    wait_after_scroll: float = 1.0,
    idle_rounds_to_stop: int = 20,
    start_date: datetime = datetime.now(TZ) - timedelta(days=2),
    end_date=None,
    mini_steps_per_round: int = 3,
    observer_timeout_ms: int = 300,
):
    driver = init_browser()

    records: list[dict] = []
    seen_urls: set[str] = set()
    consecutive_too_old_rounds = 0
    idle_rounds = 0

    if end_date is None:
        end_date = datetime.now(TZ)

    def normalize_url(u: str) -> str:
        try:
            from urllib.parse import urlsplit, urlunsplit
            s = urlsplit(u)
            path = s.path[:-1] if s.path.endswith("/") else s.path
            return urlunsplit((s.scheme, s.netloc, path, "", ""))
        except Exception:
            return u.split("#", 1)[0].split("?", 1)[0].rstrip("/")

    def process_items(items: list[dict]) -> int:
        nonlocal consecutive_too_old_rounds
        added = 0
        now = datetime.now(TZ)
        start_day = _to_local_date(start_date)
        end_day = _to_local_date(end_date)

        for it in items or []:
            href_raw = (it.get("href") or "").strip()
            if not href_raw or POST_DETAIL_KEY not in href_raw:
                continue

            href_norm = normalize_url(href_raw)
            if href_norm in seen_urls:
                continue

            tmp_date = parse_to_date(it.get("time_txt", ""), now=now)

            # ❌ Không có ngày → bỏ qua, KHÔNG reset counter
            if not tmp_date:
                continue

            # ⏹️ Bài cũ hơn start_date → tăng counter + skip
            if not (start_day <= tmp_date <= end_day):
                consecutive_too_old_rounds += 1
                continue

            seen_urls.add(href_norm)

            # Bài hợp lệ → reset counter
            consecutive_too_old_rounds = 0

            records.append({
                "url_raw": href_raw,
                "url_norm": href_norm,

                "title_raw": it.get("title", "") or "",
                "date_posted_raw": it.get("time_txt", "") or "",
                "date_posted": tmp_date,
                "card_text_raw": it.get("card_text_raw", "") or "",

                "crawl_date": now.isoformat(),

                "is_success": False,
                "error_message": "",
            })

            added += 1

        return added
    
    JS_COMMON = r"""
    const harvestOnce = () => {
    const out = [];
    const cards = Array.from(document.querySelectorAll(".mobile-card, .mantine-Card-root"));

    const toAbs = (href) => {
        if (!href) return "";
        if (href.startsWith("http")) return href;
        const a = document.createElement("a");
        a.href = href;
        return a.href;
    };

    const findDetailLink = (root) => {
        const a = root.querySelector("a[href*='/ai-news/post-detail/']");
        if (a) return a;
        return Array.from(root.querySelectorAll("a"))
        .find(x => (x.getAttribute("href") || "").includes("/ai-news/post-detail/")) || null;
    };

    for (const card of cards) {
        if (card.dataset.scraped === "1") continue;

        const linkEl = findDetailLink(card);
        if (!linkEl) continue;

        const href = toAbs(linkEl.getAttribute("href"));
        if (!href) continue;

        const title =
            (card.querySelector(".mantine-Title-root")?.innerText || "").trim();

        // ====== CHỈ TÁCH DÒNG ======
        const parts = (card.innerText || "")
            .split("\n")
            .map(x => x.trim())
            .filter(Boolean);

        const time_txt = parts.length > 5 ? parts[5] : "";
        const source   = parts.length > 7 ? parts[7] : "";

        out.push({
            href,
            title,
            time_txt,
            source,
            card_text_raw: card.innerText || "",
        });

        card.dataset.scraped = "1";
    }

    return out
    };
    """

    JS_ASYNC_HARVEST = JS_COMMON + r"""
    const scrollEl   = arguments[0];
    const stepPx     = arguments[1];
    const miniSteps  = arguments[2];
    const waitMs     = arguments[3];
    const callback   = arguments[arguments.length - 1];

    (async () => {
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));

    let batch = [];
    for (let i = 0; i < miniSteps; i++) {
        batch = batch.concat(harvestOnce());
        scrollEl.scrollTop = Math.min(scrollEl.scrollTop + stepPx, scrollEl.scrollHeight);
        await sleep(waitMs);
    }

    callback({
        items: batch,
        newHeight: scrollEl.scrollHeight
    });
    })();
    """

    try:
        # ---------- Setup ----------
        driver.get(start_url)
        driver.set_window_size(1400, 2000)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#main-scrollbar-item"))
        )
        WebDriverWait(driver, 30).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, f"a[href*='{POST_DETAIL_KEY}']")) > 0
        )

        scroll_el = driver.find_element(By.CSS_SELECTOR, "#main-scrollbar-item")
        ActionChains(driver).move_to_element(scroll_el).click(scroll_el).perform()

        prev_height = driver.execute_script(
            "return arguments[0].scrollHeight;", scroll_el
        )

        # ---------- Main loop ----------
        for _ in range(max_rounds):
            try:
                res = driver.execute_async_script(
                    JS_ASYNC_HARVEST,
                    scroll_el,
                    step_px,
                    mini_steps_per_round,
                    observer_timeout_ms,
                )
            except TimeoutException:
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight;",
                    scroll_el,
                )
                time.sleep(wait_after_scroll)
                res = {}

            items = res.get("items", []) if isinstance(res, dict) else []
            new_height = res.get("newHeight", prev_height)

            added = process_items(items)

            if added == 0 and new_height == prev_height:
                idle_rounds += 1
            else:
                idle_rounds = 0

            prev_height = new_height

            if idle_rounds >= idle_rounds_to_stop:
                break

            if consecutive_too_old_rounds >= 3:
                break

        return records

    finally:
        driver.quit()

def extract_content_selenium(driver, url):
    try:
        driver.get(url)
        content_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.content"))
        )

        html_raw = content_div.get_attribute("innerHTML")
        text_raw = content_div.text

        return {
            "is_success": True,
            "content_html_raw": html_raw,
            "content_text_raw": text_raw,
            "error_raw": "",
        }

    except Exception as e:
        return {
            "is_success": False,
            "content_html_raw": "",
            "content_text_raw": "",
            "error_raw": str(e),
        }
    
def worker_extract(
    urls,
    worker_idx=0,
):
    drv = init_browser()
    out = []

    try:
        for u in urls:
            crawl_ts = datetime.now(TZ).isoformat()

            result = extract_content_selenium(drv, u)

            logger.info(
                f"[BRONZE][Worker {worker_idx}] "
                f"url={u} | "
                f"success={result['is_success']} | "
                f"ts={crawl_ts}"
            )

            out.append({
                "url_norm": u,
                "crawl_ts": crawl_ts,
                **result,
            })

    finally:
        drv.quit()

    return out

def crawl_vietcap_news(
    url_col="url_norm",
    workers=4,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    if end_date is None:
        end_date = datetime.now(TZ)
    if start_date is None:
        start_date = datetime.now(TZ) - timedelta(days = 1)

    df = pd.DataFrame(
        discover_article_links_infinite_scroll(start_url=AI_NEWS_PAGE,
                                               start_date=start_date,
                                               end_date=end_date)
    )

    urls = df[url_col].dropna().unique().tolist()
    if not urls:
        logger.info("Không có URL để crawl.")
        return df

    shard_size = math.ceil(len(urls) / workers)
    shards = [urls[i:i+shard_size] for i in range(0, len(urls), shard_size)]

    logger.info(f"[BRONZE] Start detail crawl | urls={len(urls)}")

    results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(worker_extract, shard, idx)
            for idx, shard in enumerate(shards, start=1)
        ]

        for fut in as_completed(futs):
            try:
                results.extend(fut.result())
            except Exception as e:
                logger.exception(f"[BRONZE] Worker error: {e}")

    logger.info(f"[BRONZE] Finished detail crawl | records={len(results)}")

    # ---------- JOIN ----------
    df_detail = pd.DataFrame(
        results,
        columns=["url_norm", "content_html_raw", "content_text_raw"]
    )

    df_merged = df.merge(
        df_detail,
        on="url_norm",
        how="left"
    )

    df_merged["is_success"] = df_merged["content_html_raw"].notna()
    df_merged["error_message"] = ""

    return df_merged