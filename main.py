import pandas as pd
import numpy as np
import os

# install packages:
# pip install pandas
# pip install scipy

# working directory
os.chdir('/Users/ycx/Documents/ycx/课程/金融学概论/project')

funds = pd.read_csv('C_Fund_Return_Final.csv')
details = pd.read_csv('C_Fund_Summary_Final.csv')
funds['TradingDate'] = pd.to_datetime(funds['TradingDate'])

# risk free rate
risk_free = 0

# stock list
fund_list = list()
for symbol in details['Symbol']:
    fund = funds[funds['Symbol'] == symbol]
    ts = pd.Series(fund['ChangeRatio'].values, index=fund['TradingDate'])
    fund_list.append(ts)
fund_num = len(fund_list)

# cov matrix
cov_mat = np.zeros(shape=(fund_num, fund_num))
for i in range(fund_num):
    for j in range(fund_num):
        cov_mat[i][j] = fund_list[i].cov(fund_list[j])
        if np.isnan(cov_mat[i][j]):
            cov_mat[i][j] = 0

# mean array
mean_array = np.zeros(fund_num)
for i in range(fund_num):
    mean_array[i] = fund_list[i].mean()


def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(mean_array * weights)
    pvol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    return pret, pvol


def min_func_sharpe(weights):
    s = statistics(weights)
    return (risk_free - s[0]) / s[1]


def plot_portfolio_frontier():
    import matplotlib.pyplot as plt
    portfolio_returns = []
    portfolio_volatilities = []
    for p in range(2000):
        weights = np.random.random(fund_num)
        weights /= np.sum(weights)
        ss = statistics(weights)
        portfolio_returns.append(ss[0])
        portfolio_volatilities.append(ss[1])

    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)

    plt.figure()
    plt.scatter(portfolio_volatilities, portfolio_returns, c=(portfolio_returns - risk_free) / portfolio_volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()


def optim_weights():
    import scipy.optimize as sco
    # long only
    bnds = tuple((0, 1) for x in range(fund_num))

    # sum of weights eq to 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # optimization
    opts = sco.minimize(min_func_sharpe, np.array(fund_num * [1. / fund_num, ]), method='SLSQP', bounds=bnds, constraints=cons)

    # weights
    print('weights:')
    print(opts['x'].round(3))
    print()
    print('sharpe ratio:')
    print(-opts['fun'])  # 0.2890620311085642


def main():
    while True:
        s = input('plot portfolio frontier or weights: (0/1) ')
        if s == '0':
            plot_portfolio_frontier()
            break
        elif s == '1':
            optim_weights()
            break
        else:
            continue


main()
