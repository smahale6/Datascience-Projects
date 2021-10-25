import StrategyLearner as sl
import ManualStrategy as ms
from marketsimcode import compute_portvals
import datetime as dt
import random
import numpy as np
random.seed(1481090000)
np.random.seed(1481090000)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def author():
    return 'Shrikanth Mahale'

    
def Experiment1(symbol,sv,sd,ed):
    ########################Manual Strategy Trades##############################################
    
    manual_strategy_trades,purchase_date, sell_date  = ms.testPolicy('JPM', sd, ed, sv)
    benchmark_portvals,benchmark_CR,benchmark_mean,benchmark_std,benchmark_Sharpe_Ratio,Manual_Strategy_portvals, Manual_Strategy_CR, Manual_Strategy_mean, Manual_Strategy_std,Manual_Strategy_Sharpe_Ratio  = ms.portfolio_values(manual_strategy_trades,sv,sd,ed,symbol,impact= 0.005, commission = 9.95)
    Norm_benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]
    Norm_Manual_Strategy_portvals = Manual_Strategy_portvals / Manual_Strategy_portvals.iloc[0]
    
    ########################Strategy Learner Trades##############################################
    strategy_learner = sl.StrategyLearner(verbose=False, impact=0.005)
    strategy_learner.add_evidence(symbol, sd, ed, sv)
    strategy_learner_trades = strategy_learner.testPolicy(symbol, sd, ed, sv)
    strategy_learner_portvals, strategy_learner_CR, strategy_learner_Mean, strategy_learner_STD,strategy_learner_Sharpe_Ratio= compute_portvals(strategy_learner_trades, sv, impact=0.005,commission=9.95)
    Norm_strategy_learner_portvals = strategy_learner_portvals / strategy_learner_portvals.iloc[0]
    
    return Norm_benchmark_portvals,Norm_Manual_Strategy_portvals,Norm_strategy_learner_portvals,benchmark_CR,benchmark_mean,benchmark_std,benchmark_Sharpe_Ratio, Manual_Strategy_CR, Manual_Strategy_mean, Manual_Strategy_std,Manual_Strategy_Sharpe_Ratio,strategy_learner_CR, strategy_learner_Mean, strategy_learner_STD,strategy_learner_Sharpe_Ratio
   
def execution(symbol,sv,sd,ed):
    Norm_benchmark_portvals,Norm_Manual_Strategy_portvals,Norm_strategy_learner_portvals,benchmark_CR,benchmark_mean,benchmark_std,benchmark_Sharpe_Ratio, Manual_Strategy_CR, Manual_Strategy_mean, Manual_Strategy_std,Manual_Strategy_Sharpe_Ratio,strategy_learner_CR, strategy_learner_Mean, strategy_learner_STD,strategy_learner_Sharpe_Ratio = Experiment1(symbol,sv,sd,ed)
    ########################Plotting Manual Strategy vs Strategy Learner##############################################
    plt.figure(figsize=(7,5))
    plt.title("Manual Strategy vs. Strategy Learner")
    plt.xlabel("Dates")
    plt.ylabel("Normalized Portfolio Value")
    plt.xticks(rotation=45)
    plt.plot(Norm_Manual_Strategy_portvals, 'r', label="Manual Stategy")
    plt.plot(Norm_benchmark_portvals, 'g', label="Benchmark Stategy")
    plt.plot(Norm_strategy_learner_portvals, 'b', label="Strategy Learner")
    plt.legend()
    plt.savefig("Manual vs Strategy.png")
    plt.show()
    plt.clf()
    
    print("Strategy Learner Cumulative Returns  is {} ".format(strategy_learner_CR))
    print("Strategy Learner Average Daily Returns  is {}  ".format(strategy_learner_Mean))
    print("Strategy Learner Standard Deviation is {}  ".format(strategy_learner_STD))
    print("Strategy Learner Sharpe Ratio  is {}  ".format(strategy_learner_Sharpe_Ratio))
    
    print("Manual Strategy Cumulative Returns  is {} ".format(Manual_Strategy_CR))
    print("Manual Strategy Average Daily Returns  is {}  ".format(Manual_Strategy_mean))
    print("Manual Strategy Standard Deviation is {}  ".format(Manual_Strategy_std))
    print("Manual Strategy Sharpe Ratio  is {}  ".format(Manual_Strategy_Sharpe_Ratio))
    
    
    print("Benchmark Strategy Cumulative Returns  is {} ".format(benchmark_CR))
    print("Benchmark Strategy Average Daily Returns  is {}  ".format(benchmark_mean))
    print("Benchmark Strategy Standard Deviation is {}  ".format(benchmark_std))
    print("Benchmark Strategy Sharpe Ratio  is {}  ".format(benchmark_Sharpe_Ratio))
    
if __name__ == '__main__':
    
    symbol = 'JPM'
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    execution(symbol,sv,sd,ed)


