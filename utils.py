def date_to_season(date):
    """
    convert date string to season
    :param date: date string YYYY-MM-DD
    :return: YYYY-S
    """
    date = date[:-3]
    year = date[:4]
    month = int(date[5:7])
    season = month // 4 + 1
    return year + '-' + str(season)


def date_to_month(date):
    """
    convert date string to month
    :param date: date string YYYY-MM-DD
    :return: month string YYYY-MM
    """
    return date[:-3]


def interpolate(prev, next):
    """
    interpolate the price data
    :param prev: previous price
    :param next: next price
    :return: interpolated price
    """
    return (prev + next) / 2
