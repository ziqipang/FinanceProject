# -*- coding:utf-8 -*-
import sklearn.cluster
import numpy as np
import pandas as pd
from data_manipulation import *
import matplotlib.pyplot as plt


def mean(time_dict: dict) -> float:
    """
    calculate avg of returns
    nan is ignored

    :param time_dict: a dict of time sequence
    :return: avg of return
    """
    cnt = 0
    s = 0.0
    for k in time_dict:
        v = time_dict[k]
        if v != v:
            continue
        cnt += 1
        s += v
    return s / cnt


def cov(td1: dict, td2: dict) -> float:
    """
    calculate cov of two return sequence
    nan is ignored

    :param td1: a dict of time sequence
    :param td2: a dict of time sequence
    :return: cov
    """
    n = 0
    s1, s2, s12 = 0.0, 0.0, 0.0
    for k in td1:
        v1 = td1[k]
        if v1 != v1:
            continue
        try:
            v2 = td2[k]
        except KeyError:
            continue
        if v2 != v2:
            continue
        n += 1
        s1 += v1
        s2 += v2
        s12 += v1 * v2
    return (s12 - s1 * s2 / n) / n


def cluster(cov_mat: np.ndarray, n_clusters: int) -> None:
    # 分类，效果不好
    # linkage: {“ward”, “complete”, “average”, “single”}
    do_cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                         affinity='precomputed',
                                                         linkage='complete')
    # 计算相关性矩阵
    rho = np.zeros(shape=cov_mat.shape)
    n = cov_mat.shape[0]
    for i in range(n):
        for j in range(i, n):
            rho[i][j] = cov_mat[i][j] / np.sqrt(cov_mat[i][i] * cov_mat[j][j])
            rho[j][i] = rho[i][j]

    do_cluster.fit(1 - rho)
    print(do_cluster.labels_)
    plt.hist(do_cluster.labels_, bins=n_clusters)
    plt.show()


def grouped_mean_cov(mean_array: np.ndarray, cov_mat: np.ndarray, group_list: list) -> (np.ndarray, np.ndarray):
    """
    计算分组后的均值、协方差（组内权重相同）。
    group_list 形式: 每个元素是一组内成员的列表。
    如: [[3, 0], [1, 2], [4]]

    :param mean_array: 分组前均值
    :param cov_mat: 分组前协方差
    :param group_list: 分组
    :return: 分组后均值、协方差
    """
    group_num = len(group_list)
    ma = np.zeros(group_num)
    cm = np.zeros(shape=(group_num, group_num))
    for i in range(group_num):
        for m in group_list[i]:
            ma[i] += mean_array[m]
        ma[i] /= len(group_list[i])
    for i in range(group_num):
        for j in range(i, group_num):
            for m in group_list[i]:
                for n in group_list[j]:
                    cm[i][j] += cov_mat[m][n]
            cm[i][j] /= len(group_list[i]) * len(group_list[j])
            cm[j][i] = cm[i][j]
    return ma, cm


def optim_weights(fund_num: int, mean_array: np.ndarray, cov_mat: np.ndarray, risk_free: float, w_constraint: (float, float) = (0, 1)) -> dict:
    import scipy.optimize as sco

    def statistics(weights):
        weights = np.array(weights)
        pret = np.sum(mean_array * weights)
        pvol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
        return pret, pvol

    def min_func_sharpe(weights):
        s = statistics(weights)
        return (risk_free - s[0]) / s[1]

    # long only
    bnds = tuple(w_constraint for x in range(fund_num))

    # sum of weights eq to 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # optimization
    opts = sco.minimize(min_func_sharpe,
                        np.array(fund_num * [1. / fund_num, ]),
                        method='SLSQP', bounds=bnds, constraints=cons)

    return {
        'weights': opts['x'],
        'sharpe_ratio': -opts['fun']
    }


def risky_portion(r: float, sigma: float, A: float, risk_free: float, borrow_rate: float) -> float:
    """
    计算 portfolio 与无风险资产之间的分配

    :param r: portfolio 回报
    :param sigma: portfolio 标准差
    :param A: 衡量风险厌恶程度的系数
    :param risk_free: 无风险利率
    :param borrow_rate: 借贷利率
    :return: portfolio 分配的百分比
    """
    y = (r - risk_free) / A / sigma ** 2
    if y < 0:
        y = 0
    elif y > 1:
        y = (r - borrow_rate) / A / sigma ** 2
        if y <= 1:
            y = 1
    return y


def main():
    funds = pd.read_csv('data/C_Fund_Return_Final.csv')
    details = pd.read_csv('data/C_Fund_Summary_Final.csv')
    dates = list(funds['TradingDate'])
    calendar = sorted(list(set(dates)))

    fund_price_table = build_fund_daily_price_table(funds, details, calendar, inter=True)
    usable_funds = filter_usable_funds(fund_price_table, calendar)
    lagged_return_table = build_lagged_return_table(fund_price_table, calendar, '2015-10-01', usable_funds, 90)

    r_n = len(lagged_return_table)  # n = 46
    li = list()

    mean_vec = np.zeros(r_n)
    for i, seq in enumerate(lagged_return_table):
        li.append(seq['Symbol'])
        mean_vec[i] = mean(seq['Return'])

    # np.savetxt(os.path.join(os.getcwd(), './tmp/mean_vec.csv'), mean_vec, delimiter=',')

    cov_mat = np.zeros(shape=(r_n, r_n))
    for i in range(r_n):
        for j in range(i, r_n):
            seq1 = lagged_return_table[i]
            seq2 = lagged_return_table[j]
            cov_mat[i][j] = cov(seq1['Return'], seq2['Return'])
            cov_mat[j][i] = cov_mat[i][j]

    # np.savetxt(os.path.join(os.getcwd(), './tmp/cov_mat.csv'), cov_mat, delimiter=',')

    while True:
        s = input('cluster or sharpe ratio: (0/1) ')
        if s == '0':
            cluster(cov_mat, 15)
            break
        elif s == '1':
            res = optim_weights(r_n, mean_vec, cov_mat, 0)
            print(res['weights'].round(3))
            print(res['sharpe_ratio'])
            break
        else:
            continue


if __name__ == '__main__':
    main()
