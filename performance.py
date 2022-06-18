
## We perform the test of the best algorithm performance in terms of sharpe ratio for a rolling window
import pandas as pd

import numpy as np
from histogram_retracement import histogram_retracement
from main import stock, dataframe
from RL_algo_trading import Params_opt_algoritmic_trading, load
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

def getsharpe(data_window):
    # perform sell action based on the sign of the action
    Ret_long = data_window[f"Daily_returns_ROVI"].loc[data_window[f"Position_ROVI"] == 1]
    Ret_short = data_window[f"Daily_returns_ROVI"].loc[data_window[f"Position_ROVI"] == -1]
    returns = pd.concat([Ret_long, -Ret_short])
    sharpe = ((252 ** 0.5) * returns.mean() / returns.std())
    if np.isnan(sharpe):
        sharpe = 0
    return sharpe


def construct_memory(window ,sharpe,daily_return,action):

    memory = pd.DataFrame(columns=["k_entry", "k_exit", "EMA_days_12",
                                   "EMA_days_26", "STD_rw", "MXMN_rw",
                                   "Sharpe", "Daily_returns_ROVI", "Position_ROVI"])
    memory.loc[window,"k_entry"] = 0.85
    memory.loc[window, "k_exit"] = 0.55
    memory.loc[window, "EMA_days_12"] = 3
    memory.loc[window, "EMA_days_26"] = 30
    memory.loc[window, "STD_rw"] = 50
    memory.loc[window, "MXMN_rw"] = 26
    memory.loc[window, "Sharpe"] = sharpe
    memory.loc[window, "Daily_returns_ROVI"] = daily_return
    memory.loc[window, "Position_ROVI"] = action

    return memory

test_data = dataframe.loc[(dataframe.index > "2020-12-31")]

env_test, mean_reward,std_reward,load_model={},{},{},{}
#algs=["PPO","DDPG","A2C"]
algs=["A2C"]
#alg="A2C"
#memory_RL_allmodels= pd.DataFrame(columns=["Model", "Sum_Sharpe","Mean_Reward", "Sharpe_obtained"],
#                                  index= range(100000,605000,5000))
i=470000
for alg in algs:
    load_model[alg]=load(alg,i)
    env_test[alg] = Params_opt_algoritmic_trading(test_data, stock, type="test", model_name=alg)
    mean_reward[alg],std_reward[alg]=evaluate_policy(load_model[alg],env_test[alg],n_eval_episodes=1)
    RL_results_A2C = pd.read_csv("chooseapathyourself")
    sum_RL_A2C = RL_results_A2C.Sharpe.sum()


memory_pd= pd.DataFrame(columns=["k_entry", "k_exit", "EMA_days_12",
                              "EMA_days_26","STD_rw","MXMN_rw",
                              "Sharpe","Daily_returns_ROVI", "Position_ROVI"])

for window in range(100,len(test_data)-1):
    data_w = dataframe.iloc[window - 100:window, :]
    data_window = data_w.copy()
    #0.85,0.55,12,30,20,10
    strategy = histogram_retracement(stock=stock,
                                     dataframe=data_window,
                                     k_entry=0.85,
                                     k_exit=0.55,
                                     EMA_days_12=3,
                                     EMA_days_26=30,
                                     STD_rw=20,
                                     MXMN_rw=26)
    strategy.signal_construction()
    sharpe = getsharpe(data_window)
    action = data_window["Position_ROVI"][-1]
    daily_return = data_window["Daily_returns_ROVI"][-1]
    memory = construct_memory(window, sharpe, daily_return, action)
    memory_pd = pd.concat([memory_pd, memory])

sum_NORL = memory_pd.Sharpe.sum()
memory_pd.to_csv(f"results/values_NO_RL.csv")

plt.plot(memory_pd.Sharpe, 'b', label="Sharpes")
plt.savefig(f'results/sharpe_NO_RL.png')

results = pd.DataFrame(columns=["Sharpe NO_RL","Sharpe A2C", "Sharpe DDPG", "Sharpe PPO"])
results.loc[0,"Sharpe NO_RL"] = getsharpe(memory_pd)
results.loc[0,"Sharpe A2C"] = getsharpe(RL_results_A2C)
#results.loc[0,"Sharpe DDPG"] = getsharpe(RL_results_DDPG)
#results.loc[0,"Sharpe PPO"] = getsharpe(RL_results_PPO)


