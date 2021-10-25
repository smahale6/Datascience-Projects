import StrategyLearner as sl
from marketsimcode import compute_portvals
import datetime as dt
import random
import numpy as np
random.seed(1481090000)
np.random.seed(1481090000)
import matplotlib.pyplot as plt
import warnings
import math
warnings.filterwarnings('ignore')


def author():
    return 'Shrikanth Mahale'

def experiment2(symbol,sv,sd,ed,impact):
    strategy_learner = sl.StrategyLearner(verbose=False, impact = impact)
    strategy_learner.add_evidence(symbol, sd, ed, sv)
    strategy_learner_trades = strategy_learner.testPolicy(symbol, sd, ed, sv)
    strategy_learner_portvals, strategy_learner_CR, strategy_learner_Mean, strategy_learner_STD,strategy_learner_Sharpe_Ratio = compute_portvals(strategy_learner_trades, sv, impact,0.00)
    Norm_strategy_learner_portvals = strategy_learner_portvals / strategy_learner_portvals.iloc[0]
    return Norm_strategy_learner_portvals,strategy_learner_CR, strategy_learner_Mean, strategy_learner_STD,strategy_learner_Sharpe_Ratio

def execution(symbol,sv,sd,ed,impact_values):
    
    assert len(impact_values) == 6
    ### impact = 0.0
    impact0 = impact_values[0]
    impact0_label = 'impact = ' + str(impact0)
    Norm_strategy_learner_portvals0, strategy_learner_CR0, strategy_learner_Mean0, strategy_learner_STD0 ,strategy_learner_Sharpe_Ratio0 = experiment2(symbol,sv,sd,ed,impact0)
    
    print("Strategy Learner Cumulative Returns for impact = {} is {} ".format(impact0, strategy_learner_CR0))
    print("Strategy Learner Average Daily Returns for impact = {} is {}  ".format(impact0, strategy_learner_Mean0))
    print("Strategy Learner Standard Deviation for impact = {} is {}  ".format(impact0, strategy_learner_STD0))
    print("Strategy Learner Sharpe Ratio for impact = {} is {}  ".format(impact0, strategy_learner_Sharpe_Ratio0))
    
    ### impact = 0.025
    impact1 = impact_values[1]
    impact1_label = 'impact = ' + str(impact1)
    Norm_strategy_learner_portvals1, strategy_learner_CR1, strategy_learner_Mean1, strategy_learner_STD1 ,strategy_learner_Sharpe_Ratio1  = experiment2(symbol,sv,sd,ed,impact1)
    
    print("Strategy Learner Cumulative Returns for impact = {} is {} ".format(impact1, strategy_learner_CR1))
    print("Strategy Learner Average Daily Returns for impact = {} is {}  ".format(impact1, strategy_learner_Mean1))
    print("Strategy Learner Standard Deviation for impact = {} is {}  ".format(impact1, strategy_learner_STD1))
    print("Strategy Learner Sharpe Ratio for impact = {} is {}  ".format(impact1, strategy_learner_Sharpe_Ratio1))
    
    ### impact = 0.045
    impact2 = impact_values[2]
    impact2_label = 'impact = ' + str(impact2)
    Norm_strategy_learner_portvals2, strategy_learner_CR2, strategy_learner_Mean2, strategy_learner_STD2 ,strategy_learner_Sharpe_Ratio2  = experiment2(symbol,sv,sd,ed,impact2)
    
    print("Strategy Learner Cumulative Returns for impact = {} is {} ".format(impact2, strategy_learner_CR2))
    print("Strategy Learner Average Daily Returns for impact = {} is {}  ".format(impact2, strategy_learner_Mean2))
    print("Strategy Learner Standard Deviation for impact = {} is {}  ".format(impact2, strategy_learner_STD2))
    print("Strategy Learner Sharpe Ratio for impact = {} is {}  ".format(impact2, strategy_learner_Sharpe_Ratio2))
    
    ### impact = 0.065
    impact3 = impact_values[3]
    impact3_label = 'impact = ' + str(impact3)
    Norm_strategy_learner_portvals3, strategy_learner_CR3, strategy_learner_Mean3, strategy_learner_STD3 ,strategy_learner_Sharpe_Ratio3 = experiment2(symbol,sv,sd,ed,impact3)
    
    print("Strategy Learner Cumulative Returns for impact = {} is {} ".format(impact3, strategy_learner_CR3))
    print("Strategy Learner Average Daily Returns for impact = {} is {}  ".format(impact3, strategy_learner_Mean3))
    print("Strategy Learner Standard Deviation for impact = {} is {}  ".format(impact3, strategy_learner_STD3))
    print("Strategy Learner Sharpe Ratio for impact = {} is {}  ".format(impact3, strategy_learner_Sharpe_Ratio3))
  
    ### impact = 0.085
    impact4 = impact_values[4]
    impact4_label = 'impact = ' + str(impact4)
    Norm_strategy_learner_portvals4, strategy_learner_CR4, strategy_learner_Mean4, strategy_learner_STD4 ,strategy_learner_Sharpe_Ratio4  = experiment2(symbol,sv,sd,ed,impact4)
    
    print("Strategy Learner Cumulative Returns for impact = {} is {} ".format(impact4, strategy_learner_CR4))
    print("Strategy Learner Average Daily Returns for impact = {} is {}  ".format(impact4, strategy_learner_Mean4))
    print("Strategy Learner Standard Deviation for impact = {} is {}  ".format(impact4, strategy_learner_STD4))
    print("Strategy Learner Sharpe Ratio for impact = {} is {}  ".format(impact4, strategy_learner_Sharpe_Ratio4))
   
    ### impact = 0.1
    impact5 = impact_values[5]
    impact5_label = 'impact = ' + str(impact5)
    Norm_strategy_learner_portvals5, strategy_learner_CR5, strategy_learner_Mean5, strategy_learner_STD5 ,strategy_learner_Sharpe_Ratio5  = experiment2(symbol,sv,sd,ed,impact5)
    
    print("Strategy Learner Cumulative Returns for impact = {} is {} ".format(impact5, strategy_learner_CR5))
    print("Strategy Learner Average Daily Returns for impact = {} is {}  ".format(impact5, strategy_learner_Mean5))
    print("Strategy Learner Standard Deviation for impact = {} is {}  ".format(impact5, strategy_learner_STD5))
    print("Strategy Learner Sharpe Ratio for impact = {} is {}  ".format(impact5, strategy_learner_Sharpe_Ratio5))
  
    plt.figure(figsize=(7,5))
    plt.title("Strategy Learner Impact - Portfolio Values")
    plt.xlabel("Dates")
    plt.ylabel("Normalized Portfolio Values")
    plt.xticks(rotation=45)
    plt.plot(Norm_strategy_learner_portvals0, label=impact0_label)
    plt.plot(Norm_strategy_learner_portvals1, label=impact1_label)
    plt.plot(Norm_strategy_learner_portvals2, label=impact2_label)
    plt.plot(Norm_strategy_learner_portvals3, label=impact3_label)
    plt.plot(Norm_strategy_learner_portvals4, label=impact4_label)
    plt.plot(Norm_strategy_learner_portvals5, label=impact5_label)
    plt.legend()
    plt.savefig("Impact- Portfolio Values.png")
    plt.show()
    plt.clf()
    
    plt.figure(figsize=(7,5))
    plt.title("Strategy Learner Impact - Cumulative Returns")
    CR_dict = {'0.0':strategy_learner_CR0,'0.25':strategy_learner_CR1,'0.45':strategy_learner_CR2,'0.65':strategy_learner_CR3,'0.85':strategy_learner_CR4,'0.1':strategy_learner_CR5}
    CR_dict_keys = list(CR_dict.keys())
    CR_dict_values = list(CR_dict.values())
    plt.barh(CR_dict_keys, CR_dict_values)
    for index, value in enumerate(CR_dict_values):
        plt.text(value, index, str(round(value,2)))
    plt.axvline(0.00,color="black", linewidth = 0.5)
    plt.xlabel("Impact")
    plt.ylabel("Cumulative Returns")
    plt.savefig("Impact- Cumulative Returns.png")
    plt.show()
    plt.clf()
    
plt.show()
    
if __name__ == '__main__':
    symbol = 'JPM'
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    impact_values = [0.0,0.025,0.045,0.065,0.085,0.1]
    execution(symbol,sv,sd,ed,impact_values)

    
    

