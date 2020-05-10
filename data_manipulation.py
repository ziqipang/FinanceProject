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


def filter_usable_funds(fund_price_table, calendar, method='time'):
    """
    filter out the usable funds
    :param fund_price_table: information for funds
    :param calendar: all possible dates
    :param method: how to filter out funds, by default using time period length
    :return: symbols for usable funds
    """
    fund_symbols = list()
    for _fund in fund_price_table:
        start_date = _fund['Begin_date']
        end_date = _fund['End_date']
        period = calendar.index(end_date) - calendar.index(start_date) + 1
        if period >= 500:
            fund_symbols.append(_fund['Symbol'])

    return fund_symbols


def build_lagged_return_table(fund_price_table, calendar, start_date, usable_funds, lag_length):
    """
    return data for each usable fund in given time period
    :param fund_price_table: fund data
    :param calendar: possible dates
    :param start_date: from when
    :param usable_funds: which funds to compute
    :param lag_length: width of window size
    :return: data in similar form of fund_price table
    list [fund1, fund2, fund3, ...]
    each fund is a dictionary
    {
        'Symbol': symbol,
        'Return': dict of AvgPrice in the form of dict {date: price}
    }
    """
    lagged_return_table = list()
    for _fund in fund_price_table:
        if _fund['Symbol'] not in usable_funds:
            continue
        fund_data = dict()
        fund_data['Symbol'] = _fund['Symbol']

        # find the first usable date
        dates_in_fund = sorted(_fund['Price'].keys())
        if dates_in_fund[0] < start_date:
            first_date_index = calendar.index(start_date)
        else:
            first_date_index = calendar.index(dates_in_fund[0])

        price_data = dict()
        while first_date_index + lag_length < len(calendar) and calendar[first_date_index + lag_length] in dates_in_fund:
            begin_price = _fund['Price'][calendar[first_date_index]]
            end_price = _fund['Price'][calendar[first_date_index + lag_length]]
            return_rate = (end_price - begin_price) / (begin_price + 1e-10)
            price_data[calendar[first_date_index]] = return_rate
            first_date_index += 1

        fund_data['Return'] = price_data
        print(len(price_data.keys()))
        lagged_return_table.append(fund_data)

    return lagged_return_table


if __name__ == '__main__':
    funds, details, calendar = read_data(args)
    fund_price_table = build_fund_daily_price_table(funds, details, calendar, inter=True)
    usable_funds = filter_usable_funds(fund_price_table, calendar)
    lagged_return_table = build_lagged_return_table(fund_price_table, calendar, '2015-10-01', usable_funds, args.lag_length)

