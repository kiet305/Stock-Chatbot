from datetime import timezone, timedelta
TZ = timezone(timedelta(hours=7))
import pandas as pd
from bs4 import BeautifulSoup
from typing import Tuple, Dict

def parse_card_text_raw(card_text_raw: str) -> Tuple[str, str]:
    # 1️⃣ Tách dòng + loại bỏ dòng rỗng
    parts = [x.strip() for x in card_text_raw.splitlines() if x.strip()]

    sentiment = ""
    ticker = ""
    source = ""

    # 2️⃣ sentiment = phần tử đầu tiên
    if len(parts) >= 1:
        sentiment = parts[0]

    # 3️⃣ ticker = phần tử thứ hai
    if len(parts) >= 2:
        ticker = parts[1]

    # 4️⃣ source = phần tử ngay sau "•"
    for i, v in enumerate(parts):
        if v == "•" and i + 1 < len(parts):
            source = parts[i + 1]
            break

    return {
        "sentiment": sentiment,
        "ticker": ticker,
        "source": source,
    }

def parse_content_raw(html_raw: str) -> Dict[str, str]:
    if not html_raw:
        return {
            "summary": "",
        }

    soup = BeautifulSoup(html_raw, "html.parser")
    p_tags = soup.find_all("p")

    if not p_tags:
        return {
            "summary": "",
        }

    # ✅ tìm <p> đầu tiên có text
    main_idx = None
    for i, p in enumerate(p_tags):
        text = p.get_text(strip=True)
        if text:
            main_content = text
            main_idx = i
            break

    if main_idx is None:
        return {
            "summary": "",
        }
    
    # 2️⃣ lấy các <p> còn lại → full_content
    remaining_paragraphs = []
    for p in p_tags[main_idx + 1:]:
        text = p.get_text(strip=True)
        if text:
            remaining_paragraphs.append(text)

    full_content = "\n\n".join(remaining_paragraphs)

    return {
        "summary": main_content,
    }

def normalize_vietcap_news(df: pd.DataFrame) -> pd.DataFrame:
    card_parsed = df["card_text_raw"].apply(parse_card_text_raw).apply(pd.Series)
    content_parsed = df["content_html_raw"].apply(parse_content_raw).apply(pd.Series)
    df = df[['url_norm', 'title_raw', 'date_posted']]
    df = df.rename(columns={
        "url_norm": "url",
        "title_raw": "title",
    })

    return pd.concat(
        [
            df.drop(columns=["url_raw", "card_text_raw", "content_html_raw"], errors="ignore"),
            card_parsed,
            content_parsed,
        ],
        axis=1,
    )