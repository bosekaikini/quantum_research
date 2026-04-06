import numpy as np
import yfinance as yf


tickers = yf.tickers_sp500()

def prepare_global_returns(tickers, period="1m"):
    data = yf.download(tickers, period=period, interval="1d")['Adj Close']
    data = data.dropna(axis=1, thresh=int(len(data) * 0.8))
    data = data.ffill().bfill()
    
    returns_df = data.pct_change().dropna()
    
    #rows = stocks, columns = daily returns
    matrix = returns_df.values.T
    final_tickers = returns_df.columns.tolist()
    ticker_map = {ticker: i for i, ticker in enumerate(final_tickers)}
    
    return matrix, ticker_map

global_returns_matrix, ticker_to_index = prepare_global_returns(tickers)


def get_brightness(firefly, returns_matrix, ticker_map):
    data = firefly.get_list()
    
    # Use .get() to handle cases where a ticker might have been dropped during cleaning
    try:
        indices = [ticker_map[item[1]] for item in data]
    except KeyError:
        return -9999 # Penalty for using a ticker with no data
        
    weights = np.array([item[0] for item in data])
    sub_returns = returns_matrix[indices]
    
    # Matrix multiplication: weights (1,N) @ sub_returns (N, Days)
    port_rets = np.dot(weights, sub_returns)
    
    avg_ret = np.mean(port_rets)
    downside_rets = port_rets[port_rets < 0]
    
    if downside_rets.size == 0: 
        brightness = avg_ret * 100
    else:
        downside_std = np.sqrt(np.mean(np.square(downside_rets)))    
        brightness = avg_ret / downside_std
    
    # Penalty for weight sum != 1.0
    if abs(np.sum(weights) - 1.0) > 1e-3:
        brightness  -= 999
        
    return brightness