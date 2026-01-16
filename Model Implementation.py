import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

annual_trading_days = 252

# we will generate random w weights
Num_Portfolio = 10000
# stocks we are going to handle

#stocks = ['AAPL', 'WMT','TSLA', 'GE', 'AMZN', 'DB']

stocks = [
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS',
    'INFY.NS',
    'HINDUNILVR.NS',
    'ITC.NS',
    'LT.NS',
    'SBIN.NS',
    'BHARTIARTL.NS',
    'KOTAKBANK.NS',
    'AXISBANK.NS',
    'BAJFINANCE.NS',
    'BAJAJFINSV.NS',
    'ASIANPAINT.NS',
    'MARUTI.NS',
    'SUNPHARMA.NS',
    'TITAN.NS',
    'ULTRACEMCO.NS',
    'NTPC.NS',
    'POWERGRID.NS',
    'ONGC.NS',
    'M&M.NS',
    'WIPRO.NS',
    'HCLTECH.NS',
    'JSWSTEEL.NS',
    'TATASTEEL.NS',
    'COALINDIA.NS',
    'DIVISLAB.NS',
    'ADANIENT.NS',
    'ADANIPORTS.NS',
    'DRREDDY.NS',
    'EICHERMOT.NS',
    'HEROMOTOCO.NS',
    'BPCL.NS',
    'CIPLA.NS',
    'GRASIM.NS',
    'INDUSINDBK.NS',
    'BRITANNIA.NS',
    'HDFCLIFE.NS',
    'SBILIFE.NS',
    'APOLLOHOSP.NS',
    'UPL.NS',
    'TECHM.NS',
    'TATACONSUM.NS',
    'BAJAJ-AUTO.NS',
    'SHREECEM.NS',
    'NESTLEIND.NS'
]

#handle historical data - define start and end dates

# start_date = '2012-01-01'
# end_date = '2017-01-01'

start_date = '2020-07-01'
end_date = '2025-12-31'
risk_free_rate = 0.06


def download_data():
    #name of the stock (key) - stock values (2010-2017) as values

    stock_data = {}

    for stock in stocks:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']



    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

def calculate_return(data):
    log_return = np.log(data/data.shift(1))

    return log_return[1:]

def show_statistics(returns):
    print(returns.mean() * annual_trading_days)
    print(returns.cov() * annual_trading_days)

def show_mean_variance(returns, weights):
    #we are calculating the annual returns
    portfolio_returns = np.sum(returns.mean() * weights) * annual_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * annual_trading_days, weights)))

    print("Expected portfolio mean (return): ", portfolio_returns)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)

def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10,6))
    plt.scatter( volatilities, returns, c = returns - risk_free_rate / volatilities, marker = 'o', alpha = 0.4)
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Returns')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.show()


def generate_portfolios(returns):

    portfolio_mean = []
    portfolio_risk = []
    portfolio_weights = []
    mean_return = returns.mean() * annual_trading_days
    cov_matrix = returns.cov() * annual_trading_days

    for _ in range(Num_Portfolio):

        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_mean.append(np.sum(mean_return * w))
        portfolio_risk.append(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))


    return np.array(portfolio_weights), np.array(portfolio_mean), np.array(portfolio_risk)

def statistics(weights, returns):
    portfolio_returns = np.sum(returns.mean() * weights) * annual_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * annual_trading_days, weights)))

    return np.array([portfolio_returns, portfolio_volatility, portfolio_returns - risk_free_rate / portfolio_volatility])

#scipy opimize module can find the minimum of a given function
#the maximum of a f(x)  is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

#the constaints is sum of weights is 1
def optimize_portfolio(weights, returns):
    #the sum of weights is 1
    constraints = {'type' : 'eq', 'fun' : lambda x :np.sum(x) - 1}
    # the weights can be 1 at most : 1 when 100% of money is invested into a single stock
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun = min_function_sharpe, x0 = np.ones(len(stocks)) / len(stocks), args = returns, method = 'SLSQP', bounds = bounds, constraints= constraints)

def print_optimal_portfolio(optimum, returns):
    print("Optimal Portfolio:", optimum['x'].round(3))
    print("Expected return,  volatility and Sharpe Ratio: ", statistics(optimum['x'].round(3), returns))

def show_optimal_portfolios(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10,6))
    plt.scatter( portfolio_vols, portfolio_rets, c = portfolio_rets - risk_free_rate / portfolio_vols, marker = 'o', alpha = 0.4)
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Returns')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.plot(statistics(opt['x'],rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize = 20)
    plt.show()



if __name__ == '__main__':

    dataset = download_data().dropna()
    show_data(dataset)
    # plt.show()

    log_daily_returns = calculate_return(dataset)
    #show_statistics(log_daily_returns)

    pweights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)

    optimum = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolios(optimum, log_daily_returns, means, risks)


