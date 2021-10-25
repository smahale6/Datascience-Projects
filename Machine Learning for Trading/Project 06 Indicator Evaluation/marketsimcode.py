import pandas as pd
from util import get_data


def author():
    return 'Shrikanth Mahale'


def compute_portvals( orders, start_val = 1000000, commission=9.95, impact=0.005, symbol = 'JPM'):
    orders.sort_index(ascending=True, inplace=True)
    start_date = orders.index.min()
    end_date = orders.index.max()
    dates = pd.date_range(start_date, end_date)
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values  
    prices.drop('SPY', 1, inplace= True)
    prices['CASH'] = 1.0
    Stock_Transaction = prices.copy()
    Stock_Transaction[:] = 0

    def get_stocks(date, shares):
        tic_sym = symbol
        netholdings = Stock_Transaction[tic_sym].sum()
        if shares == 1000 and netholdings == -1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] += 2000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] -= (prices.loc[prices.index== date, tic_sym] * 2000 * (1 + impact)) + commission
        elif shares == 1000 and netholdings == 0:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] += 1000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] -= (prices.loc[prices.index== date, tic_sym] * 1000 * (1 + impact)) + commission
        elif shares == 1000 and netholdings == 1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] += 0
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] -= (prices.loc[prices.index== date, tic_sym] * 0 * (1 + impact)) + commission       
        elif shares == -1000 and netholdings == -1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] -= 0
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * 0 * (1 + impact)) + commission
        elif shares == -1000 and netholdings == 0:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] -= 1000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * 1000 * (1 + impact)) + commission
        elif shares == -1000 and netholdings == 1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] -= 2000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * 2000 * (1 + impact)) + commission
        
    orders['Date'] = orders.index    
    orders.apply(lambda x: get_stocks(x['Date'], x['Shares']), axis =1)
    
    holdings = Stock_Transaction.copy()
    holdings.loc[holdings.index == start_date, 'CASH']+= start_val
    holdings = holdings.cumsum()
    value = prices * holdings
    portval = value.sum(axis=1)
    
    dailyReturns = portval.copy()
    dailyReturns[1:] = (portval[1:] / portval[:-1].values) - 1
    dailyReturns.iloc[0] = 0
    dailyReturns = dailyReturns[1:]
    cr = (portval[-1] / portval[0]) - 1
    adr = dailyReturns.mean()
    sddr = dailyReturns.std()
    return portval, cr, adr, sddr