import time
import math
import numpy as np
import pandas as pd
import pandas_ta as ta
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime, timedelta
import requests
import zipfile
import io
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from py_vollib.black_scholes.greeks.analytical import delta
from py_vollib.black_scholes.implied_volatility import implied_volatility

# ==========================================
# 1. NSE DATA EXTRACTION
# ==========================================
def fetch_today_fno_bhavcopy():
    today = datetime.now()
    month_str = today.strftime('%b').upper()
    year_str = today.strftime('%Y')
    date_str = today.strftime('%d%b%Y').upper()
    url = f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year_str}/{month_str}/fo{date_str}bhav.csv.zip"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, timeout=10)
    response = session.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        return None
        
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            fo_df = pd.read_csv(f)
    fo_df.columns = fo_df.columns.str.strip()
    return fo_df

def extract_active_fno_universe(fo_bhavcopy_df):
    fut_stk = fo_bhavcopy_df[fo_bhavcopy_df['INSTRUMENT'] == 'FUTSTK']
    active_symbols = fut_stk['SYMBOL'].unique().tolist()
    print(f"Extracted {len(active_symbols)} active F&O stocks.")
    return active_symbols

# ==========================================
# 2. STATELESS SPOT DATA (TradingView)
# ==========================================
def fetch_universe_stateless(tickers):
    tv = TvDatafeed()
    all_data = []
    for ticker in tickers:
        try:
            df = tv.get_hist(symbol=ticker, exchange='NSE', interval=Interval.in_daily, n_bars=300)
            if df is not None and not df.empty:
                df = df.reset_index()
                df['ticker'] = ticker
                all_data.append(df)
            time.sleep(0.2)
        except Exception:
            continue
    master_df = pd.concat(all_data, ignore_index=True)
    master_df.rename(columns={'datetime': 'date'}, inplace=True)
    return master_df

# ==========================================
# 3. QUANTITATIVE ENGINE
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
    today = df[df['date'] == latest_date].copy()
    
    c_stretch = (today['RSI'] >= 80) & (today['close'] > today['BB_Upper_2.5'])
    c_momentum = today['MACD_Hist'] < 0
    c_regime = (today['ADX'] > 35) & (today['ADX_Slope'] < 0)
    c_rejection = today['Upper_Wick_Ratio'] >= 0.50
    c_premium = today['HVP'] >= 80
    c_liquidity = today['Turnover_Cr'] > 500
    
    ten_days_out = datetime.now() + timedelta(days=10)
    allowed_tickers = [t for t in today['ticker'] if earnings_calendar.get(t, datetime.max) > ten_days_out]
    c_no_earnings = today['ticker'].isin(allowed_tickers)
    
    survivors = today[c_stretch & c_momentum & c_regime & c_rejection & c_premium & c_liquidity & c_no_earnings]
    return survivors

def get_15_delta_from_bhavcopy(fo_df, ticker, spot_price):
    calls = fo_df[(fo_df['SYMBOL'] == ticker) & (fo_df['INSTRUMENT'] == 'OPTSTK') & (fo_df['OPTION_TYP'] == 'CE')].copy()
    if calls.empty: return None
    
    calls['EXPIRY_DT'] = pd.to_datetime(calls['EXPIRY_DT'])
    calls['DTE'] = (calls['EXPIRY_DT'] - datetime.now()).dt.days
    valid_expiries = calls[(calls['DTE'] >= 15) & (calls['DTE'] <= 45)]
    if valid_expiries.empty: valid_expiries = calls[calls['DTE'] > 1] 
    
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
# 4. EMAIL ALERT SYSTEM
# ==========================================
def send_email_alert(html_table):
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_APP_PASSWORD")
    recipient = os.environ.get("RECIPIENT_EMAIL")
    
    msg = MIMEMultipart()
    msg['Subject'] = f"QUANT ALERT: Short Call Candidates - {datetime.now().strftime('%d-%b-%Y')}"
    msg['From'] = sender
    msg['To'] = recipient
    
    body = f"""
    <html><body>
    <h3 style="color:#d9534f;">Actionable Naked Short Signals</h3>
    <p>Execution Rule: Abort trade manually if 9:08 AM pre-market settlement breaches the Spot Abort Level.</p>
    <table border="1" cellpadding="8" style="border-collapse: collapse; font-family: Arial;">
      <tr style="background-color: #f2f2f2;">
        <th>Ticker</th><th>Spot Close</th><th>HVP</th><th>Target Strike (15 D)</th><th>Premium</th><th>Hard Stop (200%)</th>
      </tr>
      {html_table}
    </table>
    </body></html>
    """ if html_table else "<html><body><h3>No exhaustion signals met the strict institutional criteria today.</h3></body></html>"
    
    msg.attach(MIMEText(body, 'html'))
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Fetching NSE Options Chain ---")
    fo_bhavcopy = fetch_today_fno_bhavcopy()
    
    if fo_bhavcopy is None:
        print("NSE Bhavcopy not published yet. Exiting.")
        exit()
        
    fno_universe = extract_active_fno_universe(fo_bhavcopy)
    
    print("--- Fetching TradingView History ---")
    raw_data = fetch_universe_stateless(fno_universe)
    
    print("--- Running Screener ---")
    processed = raw_data.groupby('ticker', group_keys=False).apply(compute_stateless_indicators)
    signals = execute_screener(processed, {})
    
    html_rows = ""
    if not signals.empty:
        for idx, row in signals.iterrows():
            opt = get_15_delta_from_bhavcopy(fo_bhavcopy, row['ticker'], row['close'])
            if opt:
                strike = opt['target_strike']
                prem = opt['expected_premium']
                stop = prem * 3
                html_rows += f"<tr><td><b>{row['ticker']}</b></td><td>₹{row['close']:.2f}</td><td>{row['HVP']:.1f}</td><td><b>{strike} CE</b></td><td>₹{prem:.2f}</td><td style='color:red;'><b>₹{stop:.2f}</b></td></tr>"
                print(f"Signal: {row['ticker']} | {strike} CE")
    else:
        print("No signals generated today.")
        
    print("--- Dispatching Email ---")
    send_email_alert(html_rows)
