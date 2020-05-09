import pandas as pd
import numpy as np
import os
import argparse

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--lag_length', type=int, default=90, help='the interval to compute')

args = parser.parse_args()


def read_data(args):
    """
    read the data, output funds, details and calendar
    calendar is the ordered list for all possible dates
    :param args: argument, specifying the file path
    :return: funds, details, calendar
    """
    fund_path = os.path.join(args.data_dir, 'C_Fund_Return_Final.csv')
    detail_path = os.path.join(args.data_dir, 'C_Fund_Summary_Final.csv')

    funds = pd.read_csv(fund_path)
    details = pd.read_csv(detail_path)

    dates = list(funds['TradingDate'])
    calendar = sorted(list(set(dates)))

    return funds, details, calendar


def build_fund_daily_price_table(funds, details, calendar, inter=False):
    """
    build up new data table in the form of list [fund1, fund2, fund3, ...]
    each fund is a dictionary
    {
        'Symbol': symbol,
        'Price': dict of AvgPrice in the form of dict {date: price}
        'Begin_date': date for first trade day
        'End_date': date for last trade day
    }
    :param funds: fund data
    :param details: detail data
    :param calendar: all the possible dates
    :param inter: interpolate the missing values
    :return: new data table
    """
    def interpolate_price_table(price_data, calendar, begin_date, end_date):
        """
        make up the missing value in the price table
        :param price_data: as below
        :param calendar: all the possible dates
        :return: interpolated price date
        """
        available_dates = price_data.keys()
        for _date in calendar:
            if (_date > begin_date) and (_date < end_date) and (_date not in available_dates):
                date_index = calendar.index(_date)
                prev_price = price_data[calendar[date_index - 1]]
                _i = date_index + 1
                while (calendar[_i] <= end_date) and (calendar[_i] not in available_dates):
                    _i += 1
                end_price = price_data[calendar[_i]]
                price_data[_date] = interpolate(prev_price, end_price)
        return price_data

    fund_price_table = list()
    for _symbol in details['Symbol']:
        fund_data = dict()
        fund_data['Symbol'] = _symbol
        fund = funds[funds['Symbol'] == _symbol]

        price_data = dict()
        count_trade_day = 0
        for _price, _date in zip(fund['AvgPrice'], fund['TradingDate']):
            price_data[_date] = _price
            if count_trade_day == 0:
                fund_data['Begin_date'] = _date
                count_trade_day += 1
        fund_data['End_date'] = _date

        if inter:
            price_data = interpolate_price_table(price_data, calendar, fund_data['Begin_date'], fund_data['End_date'])

        fund_data['Price'] = price_data
        fund_price_table.append(fund_data)

    return fund_price_table


def build_lagged_return_table(fund_price_table, calendar, lag_length):
    raise NotImplementedError


if __name__ == '__main__':
    funds, details, calendar = read_data(args)
    fund_price_table = build_fund_daily_price_table(funds, details, calendar, inter=True)
