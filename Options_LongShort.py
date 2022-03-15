import os
import datetime as dt
import pandas as pd
import numpy as np
import datetime
from kiteconnect import KiteConnect
import os.path
from os import path

mstrAppPath1 = "/Users/hari/Documents/Hari_Python_Projects/options-long-short-scan/Auth/"
mstrAppPath2 = "/Users/hari/Documents/Hari_Python_Projects/options-long-short-scan/Signals_Output/"
dataPath = "/Users/hari/Documents/Hari_Python_Projects/options-long-short-scan/Data/"

# OPTION_EXPIRY YY MM DD - Check format in documents
OPTION_EXPIRY = "22317"

cwd = os.chdir(mstrAppPath2)

tickers = ["NIFTY 50", "NIFTY BANK"]

# Empty Dataframe for logging signals (last step)
signal_db = pd.DataFrame(columns=["Strike", "Signal_Type", "Candle_Start_Time",
                                  "Candle_End_Time", 'Signal_Time', "Signal_Price", "New_Signal", "Timeframe"])

# generate trading session
access_token = open(mstrAppPath1 + "access_token.txt", 'r').read()
key_secret = open(mstrAppPath1 + "api_key.txt", 'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)

# get dump of all NSE instruments
instrument_dump = kite.instruments("NSE")
instrument_df = pd.DataFrame(instrument_dump)
instrument_df.to_csv(dataPath + "NSE_INST_DUMP.csv", index=False)

# get dump of all NSE Derivatives (NFO) instruments
instrument_dump_FO = kite.instruments("NFO")
instrument_df_FO = pd.DataFrame(instrument_dump_FO)
instrument_df_FO.to_csv(
    dataPath + "NSE_NFO_INST_DUMP.csv", index=False)


def log_signals(df2, _type=0):
    '''Parameters : Dataframe, Type = All records (!=0) or latest records (default value = 0)'''
    global signal_db
    if _type == 0:
        idx_start = len(df2)-1
    else:
        idx_start = 0
    for row in range(idx_start, len(df2)):
        if df2.iloc[row, 14] != 0:
            signal_db = signal_db.append({"Strike": df2.iloc[row, 13], "Signal_Type": df2.iloc[row, 14], "Candle_Start_Time": df2.iloc[row, 15], "Candle_End_Time": df2.iloc[row,
                                                                                                                                                                             15], 'Signal_Time': dt.datetime.now(), "Signal_Price": df2.iloc[row, 3], "New_Signal": "NA", "Timeframe": "5minute"}, ignore_index=True)

# Signal Generating Logic


def generate_signal(df, idx=-1):
    '''Signal is generated for the given record (idx). Parameters (Dataframe, Idx = Index) '''
    # This needs improvement. Needs to generate signals for all records.
    signal = 0
    if df.close[idx] < df.final_vwap[idx] and df.oi[idx] > df.oi_ma[idx] and df.rsi[idx] <= 49:
        signal = -1
    elif df.close[idx] > df.final_vwap[idx] and df.oi[idx] < df.oi_ma[idx] and df.rsi[idx] >= 51:
        signal = 1
    return signal

# Function to calculate VWAP


def calc_vwap(df):
    df["dt_str"] = df.index.copy()
    df["dt_str"] = df["dt_str"].apply(
        lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    df["vwap_Dtor"] = df[[
        "volume", "dt_str"]].groupby("dt_str").cumsum()

    df["hlc3_v"] = ((df["high"] + df["low"] +
                     df["close"])/3)*df["volume"]
    df["vwap_Ntor"] = df[[
        "hlc3_v", "dt_str"]].groupby("dt_str").cumsum()
    df["final_vwap"] = df["vwap_Ntor"]/df["vwap_Dtor"]

# Function to calculate RSI


def rsi(df, n):
    "function to calculate RSI"
    delta = df["close"].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[n-1]] = np.mean(u[:n])  # first value is average of gains
    u = u.drop(u.index[:(n-1)])
    d[d.index[n-1]] = np.mean(d[:n])  # first value is average of losses
    d = d.drop(d.index[:(n-1)])
    rs = u.ewm(com=n, min_periods=n).mean()/d.ewm(com=n, min_periods=n).mean()
    return 100 - 100 / (1+rs)

# Function to get strikes with defined steps and depth


def get_strikes(atm_strike, ce_pe, step=100, depth=10):
    '''CE = 1, PE = -1, DEPTH = Number of strikes above and below ATM'''
    otm_strikes = []
    for i in range(depth):
        otm_strikes.append(atm_strike+step*(i+1)*ce_pe)
    return otm_strikes

# Function to fine AT THE MONEY strike


def find_atm(current_price, round_to=100):

    price1 = int(current_price/round_to)*round_to
    price2 = price1 + round_to
    if current_price - price1 < price2 - current_price:
        selected_atm = (price1)
    else:
        selected_atm = (price2)
    return selected_atm

# Function to look up instrument token for a given script from instrument dump


def instrumentLookup(instrument_df, symbol, LookupSrc="NSE"):
    try:
        if LookupSrc == "NFO":
            return instrument_df_FO[instrument_df_FO.tradingsymbol == symbol].instrument_token.values[0]
        else:
            return instrument_df[instrument_df.tradingsymbol == symbol].instrument_token.values[0]
    except:
        return -1


# Extracts historical data and outputs in the form of dataframe
def fetchOHLC(ticker, interval, duration, LookupSrc="NSE"):
    if LookupSrc == "NFO":
        instrument = instrumentLookup(instrument_df_FO, ticker, LookupSrc)
        data = pd.DataFrame(kite.historical_data(instrument, dt.date.today(
        )-dt.timedelta(duration), dt.date.today(), interval, continuous=False, oi=True))

    else:
        instrument = instrumentLookup(instrument_df, ticker)
        data = pd.DataFrame(kite.historical_data(
            instrument, dt.date.today()-dt.timedelta(duration), dt.date.today(), interval))

    data.set_index("date", inplace=True)
    return data


# step 1: wait for 5 minute milestone, once 5 minute milestone reached, go to Step 2
current_time = datetime.datetime.now()
next_loop_time = current_time - datetime.timedelta(days=1)
next_loop_time2 = next_loop_time

while True:
    current_time = datetime.datetime.now()
    if (current_time >= next_loop_time2):
        next_loop_time2 = current_time + datetime.timedelta(seconds=11)
        print(f"AS#1:App is running, CurrentTime={current_time}")

    ############################## Section 0 ##############################
    if (current_time >= next_loop_time) and (current_time.minute % 1 == 0) and (current_time.second <= 10):
        next_loop_time = current_time + datetime.timedelta(seconds=10)
        next_loop_time = next_loop_time - \
            datetime.timedelta(seconds=next_loop_time.second)
        print("**** Inside Loop ****")

# step 2: Fetch price of Underlying Symbol

        for ticker in tickers:
            print("starting passthrough for.....", ticker)
# step 3: Determine ATM strike based of price of UL
# step 4: Choose 10 OTM strikes above ATM strike (OTM_ABOVE_01 to OTM_ABOVE_10) and 10 OTM strikes below ATM (OTM_BELOW_01 to OTM_BELOW_10)

            try:
                ohlc = fetchOHLC(ticker, "5minute", 4)
                latest_close = ohlc["close"][-1]
                if ticker == "NIFTY 50":
                    # latest_close = 15000
                    ticker_str = ticker[:5]
                    atm = find_atm(latest_close, 50)
                    print(f"Nifty AT THE MONEY : {atm}")
                    symbol_ce = get_strikes(atm, 1, 50)
                    symbol_pe = get_strikes(atm, -1, 50)
                else:
                    # latest_close = 33500
                    ticker_str = ticker[6:] + ticker[:5]
                    atm = find_atm(latest_close)
                    print(f"BankNifty AT THE MONEY : {atm}")
                    symbol_ce = get_strikes(atm, 1)
                    symbol_pe = get_strikes(atm, -1)

            except:
                print("API error for ticker :", ticker)

# Print Tickers
            call_ticker = []
            put_ticker = []

            for call in symbol_ce:
                print_strike = str(ticker_str)+OPTION_EXPIRY + str(call) + "CE"
                call_ticker.append(print_strike)
            print(call_ticker)

            for put in symbol_pe:
                print_strike = str(ticker_str)+OPTION_EXPIRY + str(put) + "PE"
                put_ticker.append(print_strike)
            print(put_ticker)

# for each of the above n number of OTM_Above strikes do following
#       step 4A: Download option price for strike OTM_Above_x for given expiry & given time-frame, past 500 bars (must include OI and VWAP)
#       step 4B: Compute VWAP and RSI on close of bar, x period MA on OI & Volume

            for ce_opt_ticker in call_ticker:
                call_data = fetchOHLC(ce_opt_ticker, "5minute", 7, "NFO")
                # print("CALL OPT TICKER DATA LEN : ", ce_opt_ticker, len(call_data))
                call_data["oi_ma"] = call_data["oi"].rolling(10).mean()
                call_data["rsi"] = rsi(call_data, 14)
                calc_vwap(call_data)
                call_data["ticker"] = ce_opt_ticker
                call_data["signal"] = generate_signal(call_data)
                call_data["Candle_start_time"] = call_data.index.copy()
                log_signals(call_data)

            for pe_opt_ticker in put_ticker:
                put_data = fetchOHLC(pe_opt_ticker, "5minute", 7, "NFO")
                # print("PUT OPT TICKER DATA LEN : ", pe_opt_ticker, len(put_data))
                put_data["oi_ma"] = put_data["oi"].rolling(10).mean()
                put_data["rsi"] = rsi(put_data, 14)
                calc_vwap(put_data)
                put_data["ticker"] = pe_opt_ticker
                put_data["signal"] = generate_signal(put_data)
                put_data["Candle_start_time"] = put_data.index.copy()
                log_signals(put_data)

        print(signal_db)

        signal_db.to_csv(mstrAppPath2 + "Signals_Log.csv",
                         mode="a", header=False)

        # exit()
