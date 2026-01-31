from google.adk.agents import Agent
from sqlalchemy import create_engine


from dotenv import load_dotenv
load_dotenv()

# Import libraries for technical agents
import yfinance as yf
import pandas as pd
import numpy as np
import re
import mplfinance as mpf
import uuid
from pathlib import Path

##### 1. Momentum Analysis Tool #####
def get_momentum(ticker: str) -> dict:
    """
    Analyze stock momentum using the RSI indicator.
    """
    print(f"--- Tool: get_momentum called for ticker: {ticker} ---")

    try:
        # Lấy dữ liệu, chỉ cần giá Close
        df = yf.download(f"{ticker}.VN", period="14d")
        if df.empty:
            return {"status": "error", "error_message": f"No data for {ticker}"}
        close = df["Close"].squeeze()

        # Tính RSI để phân tích động lượng trong 14 phiên gần nhất
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = float(rsi.dropna().iloc[-1])

        # Phân tích tính hiệu động lượng của RSI
        if latest_rsi > 70:
            rsi_signal = "RSI cho thấy cổ phiếu đang **quá mua** (overbought), có thể sắp điều chỉnh."
        elif latest_rsi < 30:
            rsi_signal = "RSI cho thấy cổ phiếu đang **quá bán** (oversold), có thể sắp hồi phục."
        else:
            rsi_signal = "RSI ở vùng trung lập, động lượng giá ổn định."

        # Tổng hợp
        interpretation = (
            f"- Chỉ báo RSI(14) hiện tại: **{latest_rsi:.2f}**\n"
            f"- Nhận định: {rsi_signal}"
        )
        return {"status": "success", "analysis": interpretation}

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

##### 2. Trend Analysis Tool #####
def get_trend(ticker: str, period: str = "3mo") -> dict:
    """
    Analyze trend using EMA, with automatic adjustment for short-, medium-, and long-term horizons based on the period.
    """
    print(f"--- Tool: get_trend called for ticker: {ticker} (period: {period}) ---")

    try:
        # Chuẩn hoá về ngày
        match = re.match(r"(\d+)(d|wk|mo|y)", period)
        if not match:
            return {"status": "error", "error_message": f"Invalid period format: {period}"}

        value, unit = int(match.group(1)), match.group(2)

        if unit == "d":
            days = value
        elif unit == "wk":
            days = value * 5
        elif unit == "mo":
            days = value * 21
        elif unit == "y":
            days = value * 252

        # Chọn xu hướng ngắn, trung, dài hạn
        if days <= 30:
            ema_fast_period, ema_slow_period = 12, 26
            horizon = "ngắn hạn"
        elif days <= 120:
            ema_fast_period, ema_slow_period = 20, 50
            horizon = "trung hạn"
        else:
            ema_fast_period, ema_slow_period = 50, 200
            horizon = "dài hạn"

        # Lấy dữ liệu
        df = yf.download(f"{ticker}.VN", period=period)
        if df.empty:
            return {"status": "error", "error_message": f"No data for {ticker}"}

        close = df["Close"].squeeze()

        # Tính EMA
        ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()

        latest_ema_fast = float(ema_fast.dropna().iloc[-1])
        latest_ema_slow = float(ema_slow.dropna().iloc[-1])

        # Phân tích xu hướng
        if latest_ema_fast > latest_ema_slow:
            trend_signal = f"Tín hiệu **Tích cực**, xu hướng tăng {horizon}."
        else:
            trend_signal = f"Tín hiệu **Tiêu cực**, xu hướng giảm {horizon}."

        # Tổng hợp
        interpretation = (
            f"- EMA({ema_fast_period}) hiện tại: **{latest_ema_fast:.2f}**\n"
            f"- EMA({ema_slow_period}) hiện tại: **{latest_ema_slow:.2f}**\n"
            f"- Khung xu hướng: **{horizon}**\n"
            f"- Nhận định: {trend_signal}"
        )

        return {"status": "success", "analysis": interpretation}

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

##### 3. Volatility Analysis Tool #####
def get_volatility(ticker: str) -> dict:
    """
    Analyze volatility using the ATR indicator (last 14 sessions)
    """
    print(f"--- Tool: get_volatility called for ticker: {ticker}---")

    try:
        # Lấy dữ liệu
        df = yf.download(f"{ticker}.VN", period="14d")
        if df.empty:
            return {"status": "error", "error_message": f"No data for {ticker}"}

        # Tính ATR
        high_low = df['High'] - df['Low']
        high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
        low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.ewm(com=14 - 1, adjust=False).mean()
        latest_atr = float(atr.dropna().iloc[-1])
        average_atr = float(atr.dropna().mean())

        # Phân tích tín hiệu
        if latest_atr > average_atr * 1.2:
            vol_signal = f"Biến động **Cao hơn** mức trung bình ({average_atr:.2f}). Giá đang dao động mạnh."
        elif latest_atr < average_atr * 0.8:
            vol_signal = f"Biến động **Thấp hơn** mức trung bình ({average_atr:.2f}). Thị trường đang chững lại."
        else:
            vol_signal = f"Biến động **Trung bình**, tương đương với giai đoạn gần đây."

        # Tổng hợp nhận định
        interpretation = (
            f"- ATR(14) hiện tại: **{latest_atr:.2f}** (biến động trung bình 14 phiên gần nhất)\n"
            f"- Nhận định biến động: {vol_signal}"
        )

        return {"status": "success", "analysis": interpretation}

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

##### 4. Volume Analysis Tool #####
def get_volume(ticker: str) -> dict:
    """
    Analyze money flow using the OBV (On-Balance Volume) indicator.
    """
    print(f"--- Tool: get_volume called for ticker: {ticker} ---")

    try:
        # Lấy dữ liệu
        df = yf.download(f"{ticker}.VN", period="2mo")
        if df.empty:
            return {"status": "error", "error_message": f"No data for {ticker}"}

        # Tính OBV
        price_change = df['Close'].diff()
        volume_direction = np.where(price_change > 0, 1,
                                    np.where(price_change < 0, -1, 0))
        directed_volume = volume_direction * df['Volume']
        obv = directed_volume.cumsum()

        # Phân tích tín hiệu
        # So sánh OBV với đường trung bình của chính nó
        obv_sma_21 = obv.rolling(window=21).mean()

        latest_obv = float(obv.dropna().iloc[-1])
        latest_obv_sma = float(obv_sma_21.dropna().iloc[-1])

        if latest_obv > latest_obv_sma:
            vol_signal = "Dòng tiền đang **Tăng** (OBV nằm trên đường trung bình của nó), xác nhận áp lực mua đang mạnh."
        else:
            vol_signal = "Dòng tiền đang **Giảm** (OBV nằm dưới đường trung bình của nó), cho thấy áp lực bán đang xuất hiện."

        # Tổng hợp
        interpretation = (
            f"- Chỉ báo OBV: {latest_obv:,.0f}\n"
            f"- Đường tín hiệu OBV (SMA 21): {latest_obv_sma:,.0f}\n"
            f"- Nhận định dòng tiền: {vol_signal}"
        )

        return {"status": "success", "analysis": interpretation}

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# ----- KHUYẾN NGHỊ -----

def suggest_ticker(
    price_col='close',
    volume_col='volume',
    short_window=15,
    long_window=30,
    vol_window=10,
    vol_ratio=1.2,
    last_n_sessions=2
):
    # --- thông tin kết nối ---
    DB_USER = "admin"
    DB_PASS = "admin123"
    DB_HOST = "localhost"
    DB_PORT = "5400"
    DB_NAME = "postgres"

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    query = """
    SELECT *
    FROM warehouse.warehouse_prices_1d
    """

    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])

    # SMA
    df['sma_short'] = df.groupby('ticker')[price_col].transform(
        lambda x: x.rolling(short_window).mean()
    )
    df['sma_long'] = df.groupby('ticker')[price_col].transform(
        lambda x: x.rolling(long_window).mean()
    )

    df['sma_short_prev'] = df.groupby('ticker')['sma_short'].shift(1)
    df['sma_long_prev'] = df.groupby('ticker')['sma_long'].shift(1)

    # Volume
    df['avg_vol_10'] = df.groupby('ticker')[volume_col].transform(
        lambda x: x.rolling(vol_window).mean()
    )

    # Cross
    golden_cross = (
        (df['sma_short'] > df['sma_long']) &
        (df['sma_short_prev'] <= df['sma_long_prev'])
    )
    death_cross = (
        (df['sma_short'] < df['sma_long']) &
        (df['sma_short_prev'] >= df['sma_long_prev'])
    )

    vol_confirm = df[volume_col] > vol_ratio * df['avg_vol_10']

    df['cross_type'] = None
    df.loc[golden_cross & vol_confirm, 'cross_type'] = 'golden_cross'
    df.loc[death_cross & vol_confirm, 'cross_type'] = 'death_cross'

    # 🔥 2 PHIÊN GIAO DỊCH MỚI NHẤT (THEO TOÀN THỊ TRƯỜNG)
    latest_sessions = (
        df['date']
        .dropna()
        .sort_values()
        .unique()
    )[-last_n_sessions:]

    result = df[
        (df['cross_type'].notna()) &
        (df['date'].isin(latest_sessions))
    ]

    buy = result.loc[result['cross_type'] == 'golden_cross', 'ticker'].tolist()
    sell = result.loc[result['cross_type'] == 'death_cross', 'ticker'].tolist()

    buy_text = "\n".join(buy) if buy else "Không có"
    sell_text = "\n".join(sell) if sell else "Không có"

    reply = (
        "Chào bạn,\n"
        "Dưới đây là các mã cổ phiếu được khuyến nghị:\n\n"
        f"📈 Cổ phiếu khuyến nghị mua:\n{buy_text}\n\n"
        f"📉 Cổ phiếu khuyến nghị bán:\n{sell_text}\n\n"
        "Kết quả dựa trên chiến lược SMA cắt qua (15, 30) "
        "và tín hiệu xác nhận từ khối lượng giao dịch."
    )
    return reply

##### 6. Aggregate Answer Tool #####
def get_answer(
    ticker: str,
    trend_result: dict,
    momentum_result: dict,
    volume_result: dict,
    volatility_result: dict
) -> str:
    """
    Aggregate analysis results from four tools (trend, momentum, volume, volatility) into a single professional response with a clear structure.
    This tool is called LAST by the Agent.
    """
    print(f"--- Tool: get_answer called to synthesize results for {ticker} ---")
    # Tự động "giải nén" nếu agent bọc nhầm
    if 'get_trend_response' in trend_result:
        trend_result = trend_result['get_trend_response']
    if 'get_momentum_response' in momentum_result:
        momentum_result = momentum_result['get_momentum_response']
    if 'get_volume_response' in volume_result:
        volume_result = volume_result['get_volume_response']
    if 'get_volatility_response' in volatility_result:
        volatility_result = volatility_result['get_volatility_response']

    # Bắt đầu xây dựng câu trả lời, sử dụng Markdown
    reply = f"Chào bạn, đây là kết quả phân tích kỹ thuật tổng hợp cho mã cổ phiếu **{ticker.upper()}**:\n\n"
    reply += "---\n\n"

    # Trend
    reply += "### Phân tích Xu hướng (Trend)\n"
    if trend_result.get("status") == "success":
        # Lấy nội dung analysis từ tool get_trend
        reply += trend_result.get("analysis", "Không có dữ liệu phân tích.") + "\n"
    else:
        # Nếu tool get_trend báo lỗi, hiển thị lỗi đó
        reply += f" *Lỗi khi phân tích xu hướng: {trend_result.get('error_message', 'Lỗi không xác định')}*\n"

    reply += "\n---\n\n" # Thêm dấu ngăn cách

    # Momentum
    reply += "###  Phân tích Động lượng (Momentum)\n"
    if momentum_result.get("status") == "success":
        reply += momentum_result.get("analysis", "Không có dữ liệu phân tích.") + "\n"
    else:
        reply += f" *Lỗi khi phân tích động lượng: {momentum_result.get('error_message', 'Lỗi không xác định')}*\n"

    reply += "\n---\n\n"

    # Volume
    reply += "###  Phân tích Khối lượng (Volume)\n"
    if volume_result.get("status") == "success":
        reply += volume_result.get("analysis", "Không có dữ liệu phân tích.") + "\n"
    else:
        reply += f" *Lỗi khi phân tích khối lượng: {volume_result.get('error_message', 'Lỗi không xác định')}*\n"

    reply += "\n---\n\n"

    # Volatility
    reply += "###  Phân tích Biến động (Volatility)\n"
    if volatility_result.get("status") == "success":
        reply += volatility_result.get("analysis", "Không có dữ liệu phân tích.") + "\n"
    else:
        reply += f" *Lỗi khi phân tích biến động: {volatility_result.get('error_message', 'Lỗi không xác định')}*\n"

    # Phần kết luận
    reply += "\n---\n\n"
    reply += "*Lưu ý: Thông tin này chỉ mang tính chất tham khảo và được tạo tự động, không phải là lời khuyên đầu tư.*"

    return reply


PROMPT = """You are a stock technical analysis expert.
Workflow:
Chose 1 of the 2 direction depend on user message
Direction 1: For analysis requests of a specific ticker:
- Call ALL four tools: get_trend, get_momentum, get_volume, get_volatility.
- Call get_answer with the results to generate the textual analysis.

Direction 2: For requests of a whole analysis for all tickers such as "which tickers should I invest":
- Call tool suggest_ticker to get the whole overview 

Output Rules:
- Display the text returned by get_answer or suggest_ticker depend on which direction
- Do NOT display raw JSON/Dictionary results from the analysis tools."""

root_agent = Agent(
    model='gemini-2.5-flash',
    name='technical_analysis',
    description='A helpful expert agent for performing stock technical analysis using various tools.',
    instruction=PROMPT,
    tools=[get_answer, get_trend, get_momentum, get_volume, get_volatility, suggest_ticker],
)
