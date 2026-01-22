import re
from typing import Dict

import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def _clean_text(node):
    """
    Lấy text tự nhiên từ một node:
    - Ưu tiên text hiển thị (anchor -> lấy .get_text()).
    - Gộp khoảng trắng, bỏ ký tự thừa (•, *, nhiều space).
    """
    txt = node.get_text(" ", strip=True)
    # Chuẩn hoá khoảng trắng
    txt = re.sub(r"\s+", " ", txt)
    # Bỏ bullet '*' đơn lẻ đầu dòng
    txt = re.sub(r"^\*\s*", "", txt)
    return txt.strip()

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

def _extract_content(soup: BeautifulSoup):
    # Lấy pHead (tiêu đề đoạn mở) nếu có
    p_head = soup.select_one('p.pHead')
    main_content = _clean_text(p_head) if p_head else ""

    # Gom tất cả pBody
    p_body_all = soup.select('p.pBody')
    parts = []
    for p in p_body_all:
        t = _clean_text(p)
        if t:
            parts.append(t)

    # Nếu chưa có main_content mà có pBody, lấy pBody[0] làm main
    if not main_content and parts:
        main_content, parts = parts[0], parts[1:]

    # Gộp phần còn lại thành detail_content
    detail_content = " ".join(parts).strip()

    return main_content, detail_content

def _extract_section(soup: BeautifulSoup) -> str:
    # đúng là "breadcrumb" (không phải "bridcrumb")
    # lấy phần tử thứ 2 trong breadcrumb nếu có
    # nhiều site dùng itemListElement cho breadcrumb
    # meta
    meta_sec = soup.select_one('meta[property="article:section"]')
    if meta_sec and meta_sec.get("content"):
        return meta_sec["content"].strip()
    return ""

def _extract_tags(soup: BeautifulSoup):
    tags = []
    tag_container = soup.find('div', class_='tags') or soup.find('div', id='tags')
    if tag_container:
        tags = [a.get_text(strip=True) for a in tag_container.find_all('a') if a.get_text(strip=True)]
    return tags

def _extract_tickers(soup: BeautifulSoup):
    tickers = []
    for div in soup.find_all("div", class_="row social_shares m-b-15"):
        # chỉ lấy nếu không có id="chisothitruong"
        if not div.has_attr("id"):
            container = div
            tickers = [a.get("title") for a in container.select("span.name-index a") if a.get("title")]
            break
    return tickers

def get_article_details(soup) -> Dict:
    main, detail = _extract_content(soup)
    tags = _extract_tags(soup)
    tickers = _extract_tickers(soup)
    section = _extract_section(soup)

    # (tuỳ chọn) lấy title nếu cần
    title = ""
    h = soup.select_one("h1, h2.post_title, h1.post_title, h1.article-title")
    if h:
        title = h.get_text(strip=True)

    return {
        "title": title,
        "tags": tags,
        "tickers": tickers,
        "summary": main,
        "full_content": detail,
        "section": section,
    }

def parse_silver_from_html(html: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")
    return get_article_details(soup)

def normalize_vietstock_news(df):
    silver_df = df[['url_norm', 'date_posted']]
    parsed = df["content_html_raw"].apply(parse_silver_from_html)
    silver_df = pd.concat(
        [
            silver_df,
            parsed.apply(pd.Series),
        ],
        axis=1,
    )
    silver_df = silver_df.rename(columns = {
        'url_norm': 'url',
    })
    silver_df = silver_df[['url', 'title', 'date_posted', 'tags', 'section', 'tickers', 'summary', 'full_content']]
    return silver_df