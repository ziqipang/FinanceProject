import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

from data_manipulation import *
from utils import *


plt.rcParams['font.family'] = 'Calibri'


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--fig_dir', type=str, default='figs/')
parser.add_argument('--eps', action='store_true', default=False, help='output eps for generating pdf')
parser.add_argument('--lag_length', type=int, default=90, help='the interval to compute')

args = parser.parse_args()


def date_occurrence_for_funds(args):
    """
    analyze the monthly occurrence pattern for funds
    :param args: argument
    :return: none
    """
    funds, details, calendar = read_data(args)
    fund_price_table = build_fund_daily_price_table(funds, details, calendar)

    months = list()
    for _date in calendar:
        months.append(date_to_season(_date))
    months = sorted(list(set(months)))

    occurrences = np.zeros(len(months))
    for _fund in fund_price_table:
        fund_occurrences = np.zeros(len(months))
        dates = _fund['Price'].keys()
        for _date in dates:
            index = months.index(date_to_season(_date))
            if fund_occurrences[index] == 0:
                fund_occurrences[index] += 1
        occurrences += fund_occurrences

    occurrences = np.array(occurrences)
    f = plt.figure()
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=10)
    plt.bar(np.arange(len(months)), occurrences, tick_label=months)
    plt.xticks(rotation=270)
    plt.xlabel('Season')
    plt.ylabel('Frequencies')
    plt.show()

    f.savefig(os.path.join(args.fig_dir, 'seasonly_occ.pdf'))


def period_length_for_funds(args):
    """
    time period for funds, draw a bar plot
    :param args: argument
    :return: none
    """
    funds, details, calendar = read_data(args)
    fund_price_table = build_fund_daily_price_table(funds, details, calendar, inter=True)

    period_length = []
    short_period_fund_symbols = []
    long_period_fund_symbols = []
    for _fund in fund_price_table:
        start_date = _fund['Begin_date']
        end_date = _fund['End_date']

        period = calendar.index(end_date) - calendar.index(start_date) + 1
        if period < 100:
            short_period_fund_symbols.append(_fund['Symbol'])
        if period > 2000:
            long_period_fund_symbols.append(_fund['Symbol'])
        period_length.append({'len': period, 'symbol': _fund['Symbol']})

    period_length = sorted(period_length, key=lambda x: x['len'])

    f = plt.figure()
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=10)

    length = []
    symbols = []
    for _item in period_length:
        length.append(_item['len'])
        symbols.append(_item['symbol'])
    plt.bar(np.arange(len(period_length)), length, tick_label=symbols, color='green')
    plt.xticks(rotation=270)
    plt.xlabel('Funds Symbols')
    plt.ylabel('Length of Available Trading Data Interval')
    plt.show()

    f.savefig(os.path.join(args.fig_dir, 'fund_period_length.pdf'))

    print(short_period_fund_symbols)
    print(long_period_fund_symbols)


if __name__ == '__main__':
    # date_occurrence_for_funds(args)
    period_length_for_funds(args)
