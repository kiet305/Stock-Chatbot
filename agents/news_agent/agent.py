from google.adk.agents.llm_agent import Agent
import numpy as np
import pandas as pd
import re
import time
from datetime import date, datetime, timedelta, timezone

from urllib.parse import urljoin, urlparse
from typing import List, Tuple, Optional, Dict
import logging
import math
from google.adk.agents.llm_agent import Agent
from src.a2a_tools.google_search import GoogleSearchAgent
from sqlalchemy import create_engine,text
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

now = datetime.now()

PROMPT = """
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
- compare to the metrics of general market by p/e or p/b of VNINDEX

DATE LOGIC FOR PRICES:
- If user specifies a date: filter that date
- If user does not specify date: return latest available date (ORDER BY date DESC LIMIT 1)
- If user specifies a range: filter by range and sort appropriately

FINAL RESPONSE REQUIREMENTS:
- NEWS: always include title + sentiment + date_posted + summary (if any) + URL link
- If user does not request a specific company: return 5 most recent market news items
"""

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
		"postgresql+psycopg2://admin:admin123@psql:5432/postgres"
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
google_search_agent = GoogleSearchAgent(agent_url=os.getenv("GOOGLE_SEARCH_AGENT_URL"))

root_agent = Agent(
    model='gemini-2.5-flash',
    name='news_agent',
    description='Agent to crawl news or stock prices from vietcap and analyze sentiment of news',
    instruction=PROMPT,
    tools=[list_columns,
           query_data,
           google_search_agent.invoke_google_search_agent_via_a2a
    ],
)