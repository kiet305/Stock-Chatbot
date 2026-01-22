from vnstock import Finance, Listing
import pandas as pd
import numpy as np

def get_stock_list():
    listing = Listing(source='VCI')

    df = pd.DataFrame(listing.symbols_by_exchange())
    df = df[df["type"].str.contains("STOCK", na=False)].reset_index(drop=True)

    df = df.rename(columns={
        "symbol": "ticker",
        "organ_short_name": "name",
        "exchange": "trading_floor"
    })

    df = df[['ticker', 'name', 'trading_floor']]

    # nhóm theo vốn hóa
    vn30 = listing.symbols_by_group('VN30')
    vn100 = listing.symbols_by_group('VN100')
    hnx30 = listing.symbols_by_group('HNX30')
    df["is_vn30"] = df["ticker"].isin(vn30)
    df["is_vn100"] = df["ticker"].isin(vn100)
    df["is_hnx30"] = df["ticker"].isin(hnx30)

    # nhóm theo ngành
    df_icb = listing.symbols_by_industries()
    df = df.merge(
        df_icb[["symbol", "icb_name2", "icb_name3"]],
        left_on="ticker",
        right_on="symbol",
        how="left"
    )
    df = df.drop(columns={'symbol'})
    df = df.rename(columns={'icb_name3': 'subindustry', 'icb_name2': 'industry'})
    
    return df

def normalize_reports(df: pd.DataFrame, report_type: str = 'is') -> list[tuple[pd.DataFrame, list[str]]]:
    df['report_type'] = df['report_type'].str.upper()
    df = df.rename(columns={
        "Năm": "year",
        "Kỳ": "quarter"
    })
    # normalize numeric
    num_cols = (
        df.select_dtypes(include=[np.number])
        .drop(columns={'year', 'quarter'}, errors="ignore")
        .columns
    )
    df[num_cols] = (df[num_cols] / 1_000_000_000).round(2)

    if report_type == 'cf':
        # --- mapping ---
        BANK_MAP = {
            "(Lãi)/lỗ các hoạt động khác": "profit_before_tax",
            "Chi từ các quỹ của TCTD": "distribution_of_funds",
            "Mua sắm TSCĐ": "fixed_assets_purchases",
            "Lưu chuyển từ hoạt động đầu tư": "cf_investment",
            "Lưu chuyển tiền từ hoạt động tài chính": "cf_finance",
            "Lưu chuyển tiền thuần trong kỳ": "net_cf",
            "Lưu chuyển tiền tệ ròng từ các hoạt động SXKD": "cf_operating",
            "Tiền và tương đương tiền cuối kỳ": "cash",
            "Cổ tức đã trả": "dividend_paid",
        }

        NON_BANK_MAP = {
            "Lãi/Lỗ ròng trước thuế": "profit_before_tax",
            "Dự phòng RR tín dụng": "risk_provision",
            "Mua sắm TSCĐ": "fixed_assets_purchases",
            "Lưu chuyển từ hoạt động đầu tư": "cf_investment",
            "Lưu chuyển tiền từ hoạt động tài chính": "cf_finance",
            "Lưu chuyển tiền thuần trong kỳ": "net_cf",
            "Lưu chuyển tiền tệ ròng từ các hoạt động SXKD": "cf_operating",
            "Tiền và tương đương tiền cuối kỳ": "cash",
            "Cổ tức đã trả": "dividend_paid"
        }
    elif report_type == 'is': 
        NON_BANK_MAP = {
            "Doanh thu thuần": "revenue",
            "Giá vốn hàng bán": "cogs",
            "Thu nhập tài chính": "fin_income",
            "Chi phí tài chính": "fin_expenses",
            "Lãi/lỗ từ công ty liên doanh": "jv_pl",
            "Chi phí bán hàng": "sales_expenses",
            "Chi phí quản lý DN": "admin_expenses",
            "Thu nhập khác": "other_income",
            "Thu nhập/Chi phí khác": "other_expenses",
            "Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)": "profit",
        }

        BANK_MAP = {
            "Doanh thu (đồng)": "revenue",
            "Thu nhập lãi và các khoản tương tự": "interest_income",
            "Chi phí lãi và các khoản tương tự": "interest_expenses",

            "Thu nhập từ hoạt động dịch vụ": "service_inccome",
            "Chi phí hoạt động dịch vụ": "service_expenses",

            "Kinh doanh ngoại hối và vàng": "fxgold_trade",
            "Chứng khoán kinh doanh": "trading_securities",
            "Chứng khoán đầu tư": "investment_securities",

            "Hoạt động khác": "other_income",
            "Chi phí hoạt động khác": "other_expense",

            "Cố tức đã nhận": "dividend_income",

            "Chi phí quản lý DN": "admin_expenses",
            "Chi phí dự phòng rủi ro tín dụng": "credit_provision",

            "Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)": "profit",
        }
    elif report_type == 'bs':
        NON_BANK_MAP = {
            "TÀI SẢN NGẮN HẠN (đồng)": "st_assets",
            "Tiền và tương đương tiền (đồng)": "cash",
            "Giá trị thuần đầu tư ngắn hạn (đồng)": "st_investment_value",
            "Các khoản phải thu ngắn hạn (đồng)": "st_receivables",
            "Hàng tồn kho ròng": "inventory",
            "Tài sản lưu động khác": "other_st_assets",

            "TÀI SẢN DÀI HẠN (đồng)": "lt_assets",
            "Phải thu về cho vay dài hạn (đồng)": "lt_receivables",
            "Tài sản cố định": "fixed_assets",
            "Giá trị ròng tài sản đầu tư": "net_asset_value",
            "Đầu tư dài hạn (đồng)": "lt_investment",
            "Tài sản dài hạn khác": "other_lt_assets",
            "Lợi thế thương mại": "goodwill",

            "TỔNG CỘNG TÀI SẢN (đồng)": "total_assets",

            "NỢ PHẢI TRẢ (đồng)": "liabilities",
            "Nợ ngắn hạn (đồng)": "current_liabilities",
            "Nợ dài hạn (đồng)": "non_current_liabilities",

            "VỐN CHỦ SỞ HỮU (đồng)": "equity",
            "Lãi chưa phân phối (đồng)": "retained_earnings",
            "LỢI ÍCH CỦA CỔ ĐÔNG THIỂU SỐ": "minority_interest",
            "Vốn góp của chủ sở hữu (đồng)": "owners_equity",

            "TỔNG CỘNG NGUỒN VỐN (đồng)": "total_equity"
        }

        BANK_MAP = {
            "TỔNG CỘNG TÀI SẢN (đồng)": "total_assets",
            "TÀI SẢN NGẮN HẠN (đồng)": "st_assets",
            "Tiền và tương đương tiền (đồng)": "cash",
            "Tiền gửi tại ngân hàng nhà nước Việt Nam": "deposit_at_SBV",
            "Tiền gửi tại các TCTD khác và cho vay các TCTD khác": "deposit_at_FI",
            "Chứng khoán kinh doanh": "trading_securities",
            "Cho vay khách hàng": "customer_loan",
            "Dự phòng rủi ro cho vay khách hàng": "risk_provision",
            "Chứng khoán đầu tư": "investment_securities",

            "Tài sản dài hạn khác (đồng)": "lt_assets",
            "Tài sản cố định (đồng)": "fixed_assets",
            "Đầu tư dài hạn (đồng)": "lt_investment",
            "Tài sản Có khác": "other_assets",

            "NỢ PHẢI TRẢ (đồng)": "liabilities",
            "Các khoản nợ chính phủ và NHNN Việt Nam": "debt_at_SBV",
            "Tiền gửi và vay các Tổ chức tín dụng khác": "debt_at_IF",
            "Tiền gửi của khách hàng": "customer_deposit",
            "Phát hành giấy tờ có giá": "securities",
            "Các khoản nợ khác": "other_debt",

            "VỐN CHỦ SỞ HỮU (đồng)": "equity",
            "Lãi chưa phân phối (đồng)": "retained_earnings",
            "LỢI ÍCH CỦA CỔ ĐÔNG THIỂU SỐ": "minority_interest",
            "Vốn góp của chủ sở hữu (đồng)": "owners_equity",
            "Quỹ của tổ chức tín dụng": "credit_fund",

            "TỔNG CỘNG NGUỒN VỐN (đồng)": "total_equity"
        }
    else:
        print("Report type must be: 'is', 'bs' or 'cf'")
        return

    # --- split bank / non-bank TRƯỚC ---
    df_list = get_stock_list()
    bank_tickers = set(df_list.loc[df_list["subindustry"] == "Ngân hàng", "ticker"])

    df_bank = df[df["ticker"].isin(bank_tickers)].copy()
    df_non_bank = df[~df["ticker"].isin(bank_tickers)].copy()

    ID_COLS = ["ticker", "year", "quarter", 'report_type']

    # --- BANK ---
    bank_cols = ID_COLS + [c for c in BANK_MAP if c in df_bank.columns]
    df_bank = (
        df_bank[bank_cols]
        .rename(columns=BANK_MAP)
        .copy()
    )

    # --- NON-BANK ---
    non_bank_cols = ID_COLS + [c for c in NON_BANK_MAP if c in df_non_bank.columns]
    df_non_bank = (
        df_non_bank[non_bank_cols]
        .rename(columns=NON_BANK_MAP)
        .copy()
    )

    return df_bank, df_non_bank

def convert_fact_table(df: pd.DataFrame) -> pd.DataFrame:
    # dimension columns (giữ cố định)
    dim_cols = ["ticker", "year", "quarter", "report_type"]

    # value columns = tất cả cột còn lại
    value_cols = [c for c in df.columns if c not in dim_cols]

    fact_df = (
        df
        .melt(
            id_vars=dim_cols,
            value_vars=value_cols,
            var_name="criteria",
            value_name="value",
        )
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )

    return fact_df
