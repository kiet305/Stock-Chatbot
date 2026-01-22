import numpy as np
import pandas as pd
import re
import time
import requests
from datetime import datetime, timedelta, timezone


from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Tuple, Optional, Dict
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("vietstock_crawler")

base_url = "https://vietstock.vn/"
headers = {"User-Agent": "Mozilla/5.0"}

def init_browser():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Khởi tạo ChromeDriver với webdriver-manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(options=options, service=service)
    return driver

def parse_time(s: str):
    """Parse '19/10/2025 17:15' hoặc 'x giây/phút/giờ/ngày trước' → datetime (giờ VN)."""
    if not s:
        return None

    s = s.strip().lower()
    now = datetime.now(timezone(timedelta(hours=7)))  # giờ Việt Nam

    # --- Trường hợp tương đối ---
    for pattern, unit in [
        (r"(\d+)\s*giây\s*trước", "seconds"),
        (r"(\d+)\s*phút\s*trước", "minutes"),
        (r"(\d+)\s*giờ\s*trước", "hours"),
    ]:
        m = re.search(pattern, s)
        if m:
            return now - timedelta(**{unit: int(m.group(1))})

    # --- Trường hợp định dạng ngày giờ tuyệt đối ---
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y, %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone(timedelta(hours=7)))
        except ValueError:
            pass

    return None

def extract_article_links(soup, base_url):
    links = []
    topic_head = soup.find('div', class_='col-lg-8 col-md-12 dual-posts sm-padding-right-30 wow fadeIn')
    articles = topic_head.select('div.single_post')
    for article in articles:
        a = article.select_one("a.fontbold") or article.select_one("h2 a, h3 a, a[href]")
        if not a:
            continue
        href = a.get("href")
        # Chuẩn hóa thành URL đầy đủ
        abs_url = href if href.startswith("http") else urljoin(base_url, href)
        links.append(abs_url)
    return links

def normalize_url(u: str) -> str:
    if not u:
        return ""
    # chuẩn hoá nhẹ: bỏ slash cuối, lower host
    try:
        p = urlparse(u.strip())
        host = p.netloc.lower()
        path = p.path.rstrip("/")
        q = f"?{p.query}" if p.query else ""
        frag = f"#{p.fragment}" if p.fragment else ""
        return f"{p.scheme}://{host}{path}{q}{frag}"
    except Exception:
        return u.strip().rstrip("/")
    
def extract_raw_time(soup: BeautifulSoup) -> tuple[str, Optional[datetime]]:
    meta = soup.select_one(
        "div.meta, div.post_meta, div.single_post_meta, div.top-meta"
    )

    if not meta:
        return "", None

    raw_time = meta.get_text(" ", strip=True)
    dt_for_stop = parse_time(raw_time)

    return raw_time, dt_for_stop

def extract_raw_article(soup: BeautifulSoup) -> Dict:
    title = ""
    h = soup.select_one("h1, h2.post_title, h1.post_title")
    if h:
        title = h.get_text(strip=True)

    time_text = ""
    meta = soup.select_one("div.meta, div.post_meta, div.single_post_meta")
    if meta:
        time_text = meta.get_text(" ", strip=True)

    return {
        "raw_title": title,
        "date_raw": time_text,
        "raw_text": soup.get_text(" ", strip=True)
    }

def crawl_vietstock_news(
    start_url: str = 'https://vietstock.vn/chu-de/1-2/moi-cap-nhat.htm',
    start_date: datetime = datetime(2025, 12, 17, tzinfo=timezone(timedelta(hours=7))),  # ⬅️ mặc định: 10/12/2025
    end_date: datetime = datetime.now(timezone(timedelta(hours=7))),                # ⬅️ ngày kết thúc
    max_pages: int = 300,
    verbose: bool = True,
) -> List[Dict]:
    """Thu thập bài viết Vietstock trong khoảng thời gian (start_date → end_date)."""
    results: List[Dict] = []
    seen_urls = set()

    try:
        driver = init_browser()
        driver.get(start_url)
        logger.info("🚀 Mở browser và bắt đầu crawl: %s", start_url)
        time.sleep(2)

        for page in range(1, max_pages + 1):
            logger.info(f"--- Trang {page} ---")
            driver.execute_script("window.scrollTo(0, 1500);")
            time.sleep(0.5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            # xử lý dữ liệu trang hiện tại
            links = extract_article_links(soup, base_url)

            for link in links:
                if "vietstock.vn" not in link:
                    continue

                if link in seen_urls:
                    continue

                is_success = False
                try:
                    resp = requests.get(link, headers=headers, timeout=15)
                    resp.raise_for_status()
                    is_success = True
                except Exception as e:
                    logger.warning("Request fail: %s", e)
                    continue

                html = resp.text
                article_soup = BeautifulSoup(html, "html.parser")

                # --- Bronze RAW extract ---
                raw_article = extract_raw_article(article_soup)
                raw_time_text, dt_for_stop = extract_raw_time(article_soup)

                # ⛔ STOP crawl nếu gặp bài cũ hơn start_date
                if dt_for_stop and dt_for_stop < start_date:
                    logger.warning(
                        "⏹️ Stop crawl – bài cũ hơn start_date: %s (%s)",
                        raw_article.get("raw_title"),
                        dt_for_stop.strftime("%d/%m/%Y")
                    )
                    return pd.DataFrame(results)

                seen_urls.add(link)

                logger.info(
                    "✓ %s | %s",
                    raw_article.get("raw_title"),
                    raw_time_text
                )

                results.append({
                    "url_raw": link,
                    "url_norm": normalize_url(link),

                    # RAW
                    "content_html_raw": html,
                    "content_text_raw": article_soup.get_text(" ", strip=True),
                    "title_raw": raw_article.get("raw_title"),
                    "date_posted_raw": raw_time_text,
                    "date_posted": parse_time(raw_time_text),

                    # crawl metadata
                    "crawl_time": datetime.now(
                        timezone(timedelta(hours=7))
                    ).isoformat(),
                    "is_success": is_success,
                })


            # sang trang kế tiếp
            if page < max_pages:
                try:
                    next_btn = WebDriverWait(driver, 30).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, 'li.pagination-page.next a'))
                    )
                    driver.execute_script("arguments[0].click();", next_btn)
                except Exception as e:
                    logger.warning(
                        "❌ Không thể sang trang kế hoặc trang không đổi → dừng crawl."
                    )
                    break

        logger.info(
            "TỔNG %d bài từ %s đến %s",
            len(results),
            start_date.strftime("%d/%m/%Y"),
            end_date.strftime("%d/%m/%Y")
        )
        return pd.DataFrame(results)

    finally:
        try:
            driver.quit()
            logger.info("Đã đóng browser")
        except Exception as e:
            logger.warning("Không đóng được browser: %s", e)