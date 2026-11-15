import time
import math
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
from datetime import date, datetime, timedelta
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from jugaad_data.nse import bhavcopy_fo_save
from vollib.black_scholes.greeks.analytical import delta
from vollib.black_scholes.implied_volatility import implied_volatility

# ==========================================
# 1. EMERGENCY FALLBACK UNIVERSE
# Only triggers if the NSE server is offline
# ==========================================
FALLBACK_FNO_UNIVERSE = [
    'AARTIIND', 'ABB', 'ABBOTINDIA', 'ABCAPITAL', 'ABFRL', 'ACC', 'ADANIENT', 'ADANIPORTS', 
    'ALKEM', 'AMBUJACEM', 'APOLLOHOSP', 'APOLLOTYRE', 'ASHOKLEY', 'ASIANPAINT', 'ASTRAL', 
    'ATUL', 'AUBANK', 'AUROPHARMA', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE', 
    'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BATAINDIA', 'BEL', 'BERGEPAINT', 'BHARATFORG', 
    'BHARTIARTL', 'BHEL', 'BIOCON', 'BOSCHLTD', 'BPCL', 'BRITANNIA', 'CANBK', 'CANFINHOME', 
    'CHAMBLFERT', 'CHOLAFIN', 'CIPLA', 'COALINDIA', 'COFORGE', 'COLPAL', 'CONCOR', 'COROMANDEL', 
    'CROMPTON', 'CUB', 'CUMMINSIND', 'DABUR', 'DALBHARAT', 'DEEPAKNTR', 'DIVISLAB', 'DIXON', 
    'DLF', 'DRREDDY', 'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL', 'GLENMARK', 
    'GMRINFRA', 'GNFC', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM', 'GUJGASLTD', 'HAL', 
    'HAVELLS', 'HCLTECH', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDCOPPER', 
    'HINDPETRO', 'HINDUNILVR', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDEA', 'IDFCFIRSTB', 
    'IEX', 'IGL', 'INDHOTEL', 'INDIACEM', 'INDIAMART', 'INDIGO', 'INDUSINDBK', 'INDUSTOWER', 
    'INFY', 'IOC', 'IPCALAB', 'IRCTC', 'ITC', 'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 
    'KOTAKBANK', 'LALPATHLAB', 'LAURUSLABS', 'LICHSGFIN', 'LT', 'LTIM', 'LTTS', 'LUPIN', 
    'M&M', 'M&MFIN', 'MANAPPURAM', 'MARICO', 'MARUTI', 'MCX', 'METROPOLIS', 'MFSL', 'MGL', 
    'MOTHERSON', 'MPHASIS', 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI', 'NAVINFLUOR', 'NESTLEIND', 
    'NMDC', 'NTPC', 'OBEROIRLTY', 'OFSS', 'ONGC', 'PAGEIND', 'PEL', 'PERSISTENT', 'PETRONET', 
    'PFC', 'PIDILITIND', 'PIIND', 'PNB', 'POLYCAB', 'POWERGRID', 'PVRINOX', 'RAMCOCEM', 
    'RBLBANK', 'RECLTD', 'RELIANCE', 'SAIL', 'SBICARD', 'SBILIFE', 'SBIN', 'SHREECEM', 
    'SIEMENS', 'SRF', 'SUNTV', 'SUNPHARMA', 'SYNGENE', 'TATACHEM', 'TATACOMM', 'TATACONSUM', 
    'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TORNTPHARM', 'TRENT', 
    'TVSMOTOR', 'UBL', 'ULTRACEMCO', 'UPL', 'VOLTAS', 'WIPRO', 'ZEEL', 'ZYDUSLIFE',
    'ANANDRATHI', 'PREMIERENE', 'LODHA', 'RADICO', 'ATHERENERG', 'PRESTIGE', 'TIINDIA', 
    'SAGILITY', 'AMBER', 'BLUESTARCO', 'BSE', 'CAMS', 'KALYANKJIL', 'KEI', 'MOTILALOFS', 
    'NYKAA', 'PATANJALI', 'SHRIRAMFIN', 'TMPV', 'UNITDSPR', 'VBL', 'ADANIENSOL', 'ADANIPOWER', 
    'ANGELONE', 'BAJAJHLDNG', 'ETERNAL', 'GODFRYPHLP', 'IREDA', 'KAYNES', 'KFINTECH', 'MAHABANK', 
    'NAM-INDIA', 'PAYTM', 'PGEL', 'PNBHOUSING', 'SOLARINDS', 'SUPREMEIND', 'TATAELXSI', 
    'UJJIVANSFB', 'APLAPOLLO', 'DMART', 'ENRIN', 'FORCEMOT', 'HYUNDAI', 'INOXWIND', 'LICI', 
    'LTF', 'MANKIND', 'MAXHEALTH', 'NBCC', 'POLICYBZR', 'UNOMINDA', 'VMM', 'ADANIGREEN', 
    'BDL', 'CDSL', 'CGPOWER', 'COCHINSHIP', 'DELHIVERY', 'GMRAIRPORT', 'IRFC', 'JIOFIN', 
    'JSWENERGY', 'LTM', 'MAZDOCK', 'OIL', 'PHOENIXLTD', 'POWERINDIA', 'RVNL', 'SONACOMS', 
    'SUZLON', 'SWIGGY', 'UNIONBANK', 'WAAREEEENER'
]

# ==========================================
# 2. NSE DATA EXTRACTION (jugaad-data)
# ==========================================
def fetch_today_fno_bhavcopy():
    check_date = date.today()
    
    for _ in range(5):
        try:
            print(f"Checking F&O Bhavcopy for {check_date}...")
            file_path = bhavcopy_fo_save(check_date, "/tmp")
            bhavcopy = pd.read_csv(file_path)
            bhavcopy.columns = bhavcopy.columns.str.strip()
            
            udiff_map = {
                "FinInstrmTp": "INSTRUMENT", "TckrSymb": "SYMBOL", "XpryDt": "EXPIRY_DT",
                "StrkPric": "STRIKE_PR", "OptnTp": "OPTION_TYP", "ClsPric": "CLOSE", "OpnIntrst": "OPEN_INT"
            }
            if "TckrSymb" in bhavcopy.columns:
                bhavcopy = bhavcopy.rename(columns=udiff_map)
            
            for col in ["INSTRUMENT", "SYMBOL", "OPTION_TYP"]:
                if col in bhavcopy.columns:
                    bhavcopy[col] = bhavcopy[col].astype(str).str.strip()
            
            bhavcopy["EXPIRY_DT"] = pd.to_datetime(bhavcopy["EXPIRY_DT"], errors='coerce')
            bhavcopy["STRIKE_PR"] = pd.to_numeric(bhavcopy["STRIKE_PR"], errors='coerce')
            bhavcopy["CLOSE"] = pd.to_numeric(bhavcopy["CLOSE"], errors='coerce')
            
            return bhavcopy
        except Exception as e:
            print(f"Bhavcopy parse failed for {check_date}: {e}")
            check_date -= timedelta(days=1)
            
    print("Could not retrieve a valid F&O Bhavcopy within the last 5 days.")
    return None

def extract_active_fno_universe(fo_bhavcopy_df):
    # Dynamically extracts all unique tickers from the live file, capturing daily additions/deletions
    active_symbols = fo_bhavcopy_df['SYMBOL'].dropna().unique().tolist()
    
    # Filter out indices mathematically
    indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX']
    active_symbols = [s for s in active_symbols if str(s).upper() not in indices and isinstance(s, str)]
    
    if 'VEDL' in active_symbols:
        active_symbols.remove('VEDL')
        
    print(f"Dynamically extracted {len(active_symbols)} active F&O stocks from the live NSE file.")
    return active_symbols

# ==========================================
# 3. STATELESS SPOT DATA (Yahoo Finance)
# ==========================================
def fetch_universe_stateless(tickers):
    print(f"Fetching 2-year history for {len(tickers)} F&O stocks...")
    yf_tickers = [f"{t}.NS" for t in tickers]
    
    data = yf.download(yf_tickers, period="2y", interval="1d", group_by='ticker', threads=True)
    
    all_data = []
    for orig_ticker, yf_ticker in zip(tickers, yf_tickers):
        if yf_ticker in data:
            df = data[yf_ticker].copy()
            if df.empty:
                continue
            df = df.dropna(subset=['Close']).reset_index()
            df = df.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            df['ticker'] = orig_ticker
            all_data.append(df)
            
    master_df = pd.concat(all_data, ignore_index=True)
    master_df.columns = master_df.columns.str.lower()
    return master_df

# ==========================================
# 4. QUANTITATIVE ENGINE
# ==========================================
def compute_stateless_indicators(group):
    group['RSI'] = ta.rsi(group['close'], length=14)
    bbands = ta.bbands(group['close'], length=20, std=2.5)
    if bbands is not None:
        group['BB_Upper_2.5'] = bbands['BBU_20_2.5']
    
    macd = ta.macd(group['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        group['MACD_Hist'] = macd['MACDh_12_26_9']
    
    group['returns'] = group['close'].pct_change()
    group['HV_20'] = group['returns'].rolling(window=20).std() * math.sqrt(252) * 100
    group['HVP'] = group['HV_20'].rolling(window=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    
    adx_df = ta.adx(group['high'], group['low'], group['close'], length=14)
    if adx_df is not None:
        group['ADX'] = adx_df['ADX_14']
        group['ADX_Slope'] = group['ADX'].diff()
        
    daily_range = np.where((group['high'] - group['low']) == 0, 0.001, group['high'] - group['low'])
    upper_wick = group['high'] - group[['open', 'close']].max(axis=1)
    group['Upper_Wick_Ratio'] = upper_wick / daily_range
    group['Turnover_Cr'] = (group['close'] * group['volume']).rolling(20).mean() / 10000000
    return group

def execute_screener(df, earnings_calendar):
    latest_date = df['date'].max()
    today = df[df['date'] == latest_date].copy().fillna(0)
    
    c_stretch = (today['RSI'] >= 80) & (today['close'] > today['BB_Upper_2.5'])
    c_momentum = today['MACD_Hist'] < 0
    c_regime = (today['ADX'] > 35) & (today['ADX_Slope'] < 0)
    c_rejection = today['Upper_Wick_Ratio'] >= 0.50
    c_premium = today['HVP'] >= 80
    c_liquidity = today['Turnover_Cr'] > 200
    
    ten_days_out = datetime.now() + timedelta(days=10)
    allowed_tickers = [t for t in today['ticker'] if earnings_calendar.get(t, datetime.max) > ten_days_out]
    c_no_earnings = today['ticker'].isin(allowed_tickers)
    
    qualifies = c_stretch & c_momentum & c_regime & c_rejection & c_premium & c_liquidity & c_no_earnings
    today['Status'] = np.where(qualifies, 'QUALIFIED', 'FAILED')
    
    today = today.sort_values(by=['Status', 'HVP'], ascending=[False, False])
    return today

def get_15_delta_from_bhavcopy(fo_df, ticker, spot_price):
    calls = fo_df[(fo_df['SYMBOL'] == ticker) & (fo_df['OPTION_TYP'] == 'CE')].copy()
    if calls.empty: return None
    
    calls['DTE'] = (calls['EXPIRY_DT'] - datetime.now()).dt.days
    valid_expiries = calls[(calls['DTE'] >= 15) & (calls['DTE'] <= 45)]
    if valid_expiries.empty: valid_expiries = calls[calls['DTE'] > 1] 
    
    if valid_expiries.empty: return None
    
    target_expiry = valid_expiries['EXPIRY_DT'].min()
    chain = valid_expiries[valid_expiries['EXPIRY_DT'] == target_expiry].copy()
    target_dte = chain['DTE'].iloc[0]
    
    options_data = []
    for _, row in chain.iterrows():
        strike = row['STRIKE_PR']
        opt_price = row['CLOSE']
        if strike <= spot_price or opt_price <= 0.05: continue
        try:
            iv = implied_volatility(opt_price, spot_price, strike, target_dte / 365.0, 0.07, 'c')
            opt_delta = delta('c', spot_price, strike, target_dte / 365.0, 0.07, iv)
            options_data.append({'strike': strike, 'premium': opt_price, 'delta': opt_delta})
        except Exception:
            continue
            
    if not options_data: return None
    df_chain = pd.DataFrame(options_data)
    df_chain['delta_distance'] = abs(df_chain['delta'] - 0.15)
    best_strike = df_chain.loc[df_chain['delta_distance'].idxmin()]
    
    return {
        'target_strike': best_strike['strike'],
        'expected_premium': best_strike['premium']
    }

# ==========================================
# 5. EMAIL ALERT SYSTEM
# ==========================================
def send_email_alert(html_table):
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_APP_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL")
    
    msg = MIMEMultipart()
    msg['Subject'] = f"QUANTITATIVE EOD MATRIX: Naked Short Universe - {date.today().strftime('%d-%b-%Y')}"
    msg['From'] = sender
    msg['To'] = recipient
    
    body = f"""
    <html>
    <head>
    <style>
      table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }}
      th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
      th {{ background-color: #0b1c3c; color: white; text-align: center; }}
      td:first-child {{ text-align: left; font-weight: bold; }}
      .qualified {{ background-color: #d4edda; }}
      .failed {{ background-color: #ffffff; color: #555; }}
    </style>
    </head>
    <body>
    <h2 style="color:#0b1c3c;">Systematic F&O Exhaustion Matrix</h2>
    <p>Displays all mathematically processed F&O constituents. Sorted by Signal Status, then HVP. <br>
    <b>Rule:</b> Only execute 'QUALIFIED' tickers. Abort if 9:08 AM pre-market breaches Spot Close.</p>
    
    <table>
      <tr>
        <th>Ticker</th>
        <th>Spot Close</th>
        <th>RSI (14)</th>
        <th>MACD Hist</th>
        <th>ADX</th>
        <th>Wick %</th>
        <th>HVP (252)</th>
        <th>Liq (Cr)</th>
        <th>Target (15D)</th>
        <th>Premium</th>
        <th>Hard Stop</th>
        <th>Signal Status</th>
      </tr>
      {html_table}
    </table>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
    except Exception as e:
        print(f"CRITICAL: Email delivery failed: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Fetching NSE Options Chain ---")
    fo_bhavcopy = fetch_today_fno_bhavcopy()
    
    if fo_bhavcopy is not None:
        # Dynamically auto-detects the entire F&O list today
        fno_universe = extract_active_fno_universe(fo_bhavcopy)
    else:
        print("NSE Options Data Blocked or Unavailable. Activating Fallback F&O Universe.")
        fno_universe = [ticker for ticker in FALLBACK_FNO_UNIVERSE if ticker != 'VEDL']
        
    print("--- Fetching Yahoo Finance History ---")
    raw_data = fetch_universe_stateless(fno_universe)
    
    print("--- Running Screener ---")
    processed_list = []
    for ticker_name, group_df in raw_data.groupby('ticker'):
        processed_list.append(compute_stateless_indicators(group_df.copy()))
    processed = pd.concat(processed_list, ignore_index=True)
    
    full_market_matrix = execute_screener(processed, {})
    
    html_rows = ""
    print("--- Calculating Greeks & Building Matrix ---")
    
    for idx, row in full_market_matrix.iterrows():
        strike_str = "N/A"
        prem_str = "N/A"
        stop_str = "N/A"
        
        if fo_bhavcopy is not None:
            opt = get_15_delta_from_bhavcopy(fo_bhavcopy, row['ticker'], row['close'])
            if opt:
                strike_str = f"{opt['target_strike']} CE"
                prem_str = f"₹{opt['expected_premium']:.2f}"
                stop_str = f"₹{opt['expected_premium'] * 3:.2f}"
            else:
                strike_str = "Illiquid Chain"
        elif row['Status'] == 'QUALIFIED':
            strike_str = "Manual Pull Required"
            
        row_class = "qualified" if row['Status'] == 'QUALIFIED' else "failed"
        status_color = "green" if row['Status'] == 'QUALIFIED' else "red"
        
        html_rows += f"""
        <tr class="{row_class}">
            <td>{row['ticker']}</td>
            <td>₹{row['close']:.2f}</td>
            <td>{row['RSI']:.1f}</td>
            <td>{row['MACD_Hist']:.2f}</td>
            <td>{row['ADX']:.1f}</td>
            <td>{row['Upper_Wick_Ratio']:.2f}</td>
            <td>{row['HVP']:.1f}</td>
            <td>₹{row['Turnover_Cr']:.1f}</td>
            <td style="font-weight:bold;">{strike_str}</td>
            <td>{prem_str}</td>
            <td style="color:red; font-weight:bold;">{stop_str}</td>
            <td style="color:{status_color}; font-weight:bold;">{row['Status']}</td>
        </tr>
        """
        
    print("--- Dispatching Executive Report ---")
    send_email_alert(html_rows)
    print("Complete.")
