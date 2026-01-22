from google.adk.agents.llm_agent import Agent
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
from google.adk.agents.llm_agent import Agent
from sqlalchemy import create_engine,text

import logging
logger = logging.getLogger("vietcap_bronze")
logger.setLevel(logging.INFO)

BASE = "https://trading.vietcap.com.vn"
AI_NEWS_PAGE = "https://trading.vietcap.com.vn/ai-news/market"
POST_DETAIL_KEY = "/ai-news/post-detail/"
TZ = timezone(timedelta(hours=7))

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
DRIVER_PATH = ChromeDriverManager().install()

def discover_article_links_infinite_scroll():
    max_rounds= 200
    step_px= 2000
    wait_after_scroll= 1.0
    idle_rounds_to_stop= 20
    mini_steps_per_round= 3
    observer_timeout_ms= 600
    end_date = datetime.now(TZ)
    start_date = datetime.now(TZ)

    start_url=AI_NEWS_PAGE
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(DRIVER_PATH), options=options)


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

        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#main-scrollbar-item"))
        )
        WebDriverWait(driver, 60).until(
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
 
def crawl_vietcap_news():
    """Crawl latest news from vietcap website."""

    df = pd.DataFrame(
        discover_article_links_infinite_scroll()
    )
    df['sentiment'] = df['card_text_raw'].str.split('\n').str[0]
    df.drop(columns=['card_text_raw','url_raw','date_posted_raw','crawl_date','is_success','error_message'], inplace=True)
    df['company ticker'] = df['title_raw'].str.split(':').str[0]
    return df.to_json(orient="records")

from datetime import date, datetime
from typing import Any

def json_safe(obj: Any) -> Any:
    """Convert objects (date/datetime/np types) into JSON serializable values."""
    # datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # numpy types (optional but safe)
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass

    # dict/list/tuple
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [json_safe(x) for x in obj]

    return obj

PROMPT = """"
    You are an Expert Agent designed to help users query and analyze data stored in a PostgreSQL database,
    and retrieve the latest Vietcap-related news.

    IMPORTANT RULE:
    - Always map company name -> stock ticker first (e.g., Vietcombank -> VCB).
    - Use the ticker for all next steps.

    AVAILABLE TOOLS:
    - list_columns: inspect table structure (MUST be called first)
    - query_data: execute SQL queries safely on the Postgres database

    DATABASE TABLES:
    1) warehouse.warehouse_news
    Columns: url, title, summary, tags, section, sentiment, date_posted

    2) warehouse.warehouse_prices_1d
    Contains daily stock prices, OHLCV, and optional metrics like P/E, P/B, chg_1d, chg_1w, chg_1m,...

    MANDATORY TOOL PIPELINE:
    - If user asks about news/current events/updates -> MUST:
    (1) list_columns
    (2) query_data

    - If user asks about prices/valuation/price changes -> MUST:
    (1) list_columns
    (2) query_data

    MANDATORY DATA FILTERS FOR NEWS (always apply):
    - title IS NOT NULL
    - url IS NOT NULL

    USER INTENT CLASSIFICATION (mandatory):
    Classify the query into one of these:

    A) GENERAL / MARKET-WIDE TOPIC
    Examples:
    - "thị trường hôm nay thế nào?"
    - "kinh tế vĩ mô"
    - "chính sách nhà nước"
    - "lãi suất", "tỷ giá", "lạm phát", "GDP"
    - "ngân hàng", "bất động sản", "chứng khoán"

    B) COMPANY-SPECIFIC TOPIC
    Examples:
    - "tin về VCB"
    - "FPT có gì mới?"
    - "cập nhật HPG"
    - company name that can be mapped to a ticker

    TIME FILTERING (mandatory if user mentions date/period):
    If user mentions a time range ("hôm nay", "7 ngày qua", "tuần này", "từ A đến B"):
    - filter by date_posted within that period
    - return newest items first

    ==================================================
    A) GENERAL / MARKET-WIDE TOPIC RULES
    ==================================================
    Goal: return news representing the topic clearly via title and section/tags.

    Ranking priority:
    1) title contains topic keywords (highest)
    2) section/tags contains topic keywords (if not null)
    3) newest date_posted

    Sentiment is NOT mandatory for GENERAL topic.

    GENERAL RETRIEVAL STRATEGY (multi-tier):
    Tier G1 (best): title/summary match + section/tags match if available
    Tier G2 (fallback): title/summary match only
    Tier G3 (broad expansion): expand keywords for broad topics:
    - "chính sách nhà nước" -> ["nghị định","thông tư","chính phủ","quốc hội","bộ","cải cách","điều hành","chỉ thị"]
    - "kinh tế vĩ mô" -> ["GDP","CPI","lạm phát","lãi suất","tỷ giá","tăng trưởng","xuất khẩu","FDI","vĩ mô"]
    - "thị trường chung" -> ["VN-Index","chứng khoán","thị trường","dòng tiền","khối ngoại","tâm lý","thanh khoản"]
    Tier G4 (guarantee >= 3 results): allow section/tags/sentiment NULL but keep title/url NOT NULL

    GENERAL OUTPUT REQUIREMENTS:
    Return at least 3 news items.
    Each item must include:
    - title
    - section/tags (if available)
    - summary (if available)
    - date_posted (if available)
    - url
    - sentiment (if available)

    ==================================================
    B) COMPANY-SPECIFIC TOPIC RULES
    ==================================================
    Goal: return company news with explicit sentiment and company keyword in title.

    Mandatory preferences:
    - title must contain ticker OR company name
    - prefer sentiment IS NOT NULL

    COMPANY RETRIEVAL STRATEGY (multi-tier):
    Tier C1 (best): sentiment NOT NULL + title contains ticker/company name
    Tier C2 (fallback): sentiment NOT NULL + summary contains ticker/company name
    Tier C3 (fallback): allow sentiment NULL but title must contain ticker/company name
    Tier C4 (guarantee >= 5 results): search tags/section contains ticker/company name, rank sentiment NOT NULL first, then newest

    COMPANY OUTPUT REQUIREMENTS:
    Return at least 5 news items.
    Each item must include:
    - title (must contain company keyword)
    - sentiment (prefer not null)
    - date_posted (if available)
    - summary (if available)
    - section/tags (if available)
    - url

    EMPTY RESULT HANDLING (mandatory):
    If a query returns empty:
    - retry immediately using the next tier
    - do NOT answer generically without attempting all tiers

    ==================================================
    PRICES RULES (Normalized)
    ==================================================

    ASSET TYPE DETECTION:
    - If user asks about VN30, VN100, HNX30, UPCOMINDEX, VNINDEX:
    treat them as INDEX (not company ticker)

    - Otherwise treat as STOCK TICKER (after mapping company name -> ticker)

    PRICES OUTPUT (OHLCV):
    When user asks "giá", "giá hôm nay", "giá ngày X", "OHLCV":
    - query open, high, low, close, volume (+ date)

    PRICE CHANGE METRICS (chg_*):
    If user asks "tăng giảm 1 ngày/1 tuần/1 tháng...":
    - use columns: chg_1d, chg_1w, chg_1m, chg_3m, chg_6m, chg_1y (if available)
    - MUST remove null values: only return rows where requested chg_* IS NOT NULL

    VALUATION METRICS (P/E, P/B):
    If user asks about P/E or P/B:
    - query pe and pb (or equivalent columns)
    - MUST remove null values:
    - pe IS NOT NULL for P/E
    - pb IS NOT NULL for P/B

    DATE LOGIC FOR PRICES:
    - If user specifies a date: filter that date
    - If user does not specify date: return latest available date (ORDER BY date DESC LIMIT 1)
    - If user specifies a range: filter by range and sort appropriately

    FINAL RESPONSE REQUIREMENTS:
    - NEWS: always include title + sentiment + date_posted + summary (if any) + URL link
    - If user does not request a specific company: return 5 most recent market news items
    """

def query_data(sql: str):
    """Execute SQL queries to get news safely on the Postgres database."""
    engine = create_engine(
        "postgresql+psycopg2://admin:admin123@psql:5432/postgres"
    )
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()

        data = [dict(r._mapping) for r in rows]
        return json_safe(data)   # ✅ FIX: convert date -> string before returning

    except Exception as e:
        return {"error": str(e)}  # ✅ return JSON-safe object instead of raw string

def list_columns():
    """Execute SQL queries query of columns name and data type in specific table that need to search."""
    engine = create_engine(
        "postgresql+psycopg2://admin:admin123@psql:5432/postgres"
    )
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'warehouse'
                  AND table_name = 'warehouse_news'
                ORDER BY ordinal_position
            """))
            rows = result.fetchall()

        data = [dict(r._mapping) for r in rows]
        return json_safe(data)

    except Exception as e:
        return {"error": str(e)}

def list_table_from_stock() -> str:
    """List all tables from stock schema, help for chatbot to know which table to query."""
    engine = create_engine(
		"postgresql+psycopg2://admin:admin123@localhost:5460/postgres"
	)
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'warehouse' ORDER BY table_name"))
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()     

root_agent = Agent(
    model='gemini-2.5-flash',
    name='news_agent',
    description='Agent to crawl news from vietcap and analyze sentiment of news',
    instruction=PROMPT,
    tools=[crawl_vietcap_news,list_columns,query_data],
)

