
from main import dataframe, stock
from stable_baselines3.common.evaluation import evaluate_policy

from RL_algo_trading import Params_opt_algoritmic_trading, train, load



#Training data:(2007-12-06 ->2017-01-01)
train_data = dataframe.loc[(dataframe.index >= "2007-12-06") & (dataframe.index < "2018-12-31")]
#validation data:(2018-12-31 ->2020-01-01)
val_data = dataframe.loc[(dataframe.index > "2018-12-31") & (dataframe.index <= "2020-12-31")]
# test data: (2020-01-01 -> now)
test_data = dataframe.loc[(dataframe.index > "2020-12-31")]

algorithm_list=["A2C", "DDPG", "PPO"]
##TRAINING RL

env_test, env_val, env_train, model, load_model, mean_reward,std_reward={},{},{},{},{},{},{}

for alg in algorithm_list:
    ##Defining Environments
    env_train[alg] = Params_opt_algoritmic_trading(train_data, stock, type="train", model_name=alg)
    env_val[alg] = Params_opt_algoritmic_trading(val_data, stock, type="val", model_name=alg)
    env_test[alg] = Params_opt_algoritmic_trading(test_data, stock, type="test", model_name=alg)

    ##Training Starts
    print(f"======Training {alg} Starts=======")
    #for i in range(1,10):
    model[alg]=train(alg, env_train[alg], f"training", timesteps=600000)
    print(f"======Training {alg} Finish=======")
    load_model[alg]=load(alg)
    ##VALIDATION
    print(f"======Validation {alg} Starts=======")
    #score[alg]=test_validation(5,env_val[alg],load_model[alg])
    mean_reward[alg],std_reward[alg]=evaluate_policy(load_model[alg],env_val[alg],n_eval_episodes=10)
    print(f"======Validation {alg} Finish=======")


