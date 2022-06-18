from gym.utils import seeding
from gym import spaces
import pandas as pd
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
from histogram_retracement import histogram_retracement
from main import stock

# RL models from stable-baselines
from stable_baselines3 import A2C, DDPG,PPO
from stable_baselines3.common.callbacks import CheckpointCallback


logdir="./Logs"
model_path="./Models/"
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=model_path, name_prefix="rl_algo")
def train(algorithm, env_train, model_name, timesteps=25000):
    if algorithm == "A2C":
        """A2C model"""
        start = time.time()
        model = A2C('MlpPolicy', env_train, verbose=1, tensorboard_log=logdir)
        model.learn(total_timesteps=timesteps, tb_log_name="A2C", callback=checkpoint_callback)
        end = time.time()

        model.save(f"Training/A2C_{model_name}")
        print('Training time (A2C): ', (end - start) / 60, ' minutes')
        return model
    if algorithm == "DDPG":
        """DDPG model"""
        start = time.time()
        model = DDPG('MlpPolicy', env_train, verbose=1, tensorboard_log=logdir)
        model.learn(total_timesteps=timesteps,tb_log_name="DDPG")
        end = time.time()

        model.save(f"Training/DDPG_{model_name}")
        print('Training time (DDPG): ', (end - start) / 60, ' minutes')
        return model
    if algorithm == "PPO":
        """PPO model"""
        start = time.time()
        model = PPO('MlpPolicy', env_train, verbose=1 ,ent_coef=0.005, tensorboard_log=logdir)
        model.learn(total_timesteps=timesteps, tb_log_name="PPO")
        end = time.time()
        model.save(f"Training/PPO_{model_name}")
        print('Training time (PPO): ', (end - start) / 60, ' minutes')
        return model



def load(algorithm,i):
    if algorithm == "A2C":
        """A2C model"""
        model = A2C.load(f"./rl_algo_{str(i)}_steps.zip")
        return model
    if algorithm == "DDPG":
        """DDPG model"""
        model = DDPG.load(f"./Training/DDPG_training.zip")
        return model
    if algorithm == "PPO":
        """PPO model"""
        model = PPO.load(f"./Training/PPO_training.zip")
        return model



class Params_opt_algoritmic_trading(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, dataframe, stock, window=100, type="train", model_name="A2C"):
        # Preprocessing of Dataframe
        self.window = window
        self.stock = stock
        self.type = type  # val, train or test
        self.model_name = model_name
        self.dataframe = dataframe
        # actions in the 2 first parameters of the histogram retracemnt strategy.
        #Although they are scaled between -1 and 1 the real values are between 0,5 and 0.95
        self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # load data from a pandas dataframe
        data_win = self.dataframe.iloc[self.window - 100:self.window, :]
        self.data_window_RL = data_win.copy() #I do this to avoid slicing problems
        self.data_window_NO_RL = data_win.copy()
        self.terminal = False
        # initalize state
        self.state = self.reset()

        # initialize reward
        self.reward = 0

        # memorize things
        self._seed()

    def getsharpe(self, data_window):
        """Method that calculates the sharpe for a given dataframe"""
        Ret_long = data_window[f"Daily_returns_{self.stock}"].loc[data_window[f"Position_{self.stock}"] == 1]
        Ret_short = data_window[f"Daily_returns_{self.stock}"].loc[data_window[f"Position_{self.stock}"] == -1]
        returns=pd.concat([Ret_long,-Ret_short])
        sharpe = ((252 ** 0.5) * returns.mean() / returns.std())
        if np.isnan(sharpe):
            sharpe=0
        return sharpe

    def minmaxscaler_actions(self, action_scaled):
        """As it is good practices to scale actions for algorithms between -1 and 1, those actions are
        rescaled at the original objective
        k_entry:       X1,X2[-1,1] -->Y1,Y2[0.75, 0.99]
        k_exit:        X1,X2[-1,1] -->Y1,Y2[0.5,0.75]
        EMA_days_12:   X1,X2[-1,1] -->Y1,Y2[3,12]
        EMA_days_26:   X1,X2[-1,1] -->Y1,Y2[26,50]
        STD_rw:        X1,X2[-1,1] -->Y1,Y2[20,50]
        MXMN_rw:       X1,X2[-1,1] -->Y1,Y2[10,50]
        """
        X1, X2 = -1, 1
        Y1 = [0.75, 0.5, 3, 26, 20, 10]
        Y2 = [0.99, 0.75, 12, 50, 50, 50]
        act_k_entry = ((action_scaled[0] - X1) / (X2 - X1) * (Y2[0] - Y1[0])) + Y1[0]
        act_k_exit = ((action_scaled[1] - X1) / (X2 - X1) * (Y2[1] - Y1[1])) + Y1[1]
        act_EMA_12 = ((action_scaled[2] - X1) / (X2 - X1) * (Y2[2] - Y1[2])) + Y1[2]
        act_EMA_26 = ((action_scaled[3] - X1) / (X2 - X1) * (Y2[3] - Y1[3])) + Y1[3]
        act_STD_rw = ((action_scaled[4] - X1) / (X2 - X1) * (Y2[4] - Y1[4])) + Y1[4]
        act_MXMN_rw = ((action_scaled[5] - X1) / (X2 - X1) * (Y2[5] - Y1[5])) + Y1[5]
        action_unscaled = np.array([act_k_entry,
                                    act_k_exit,
                                    int(act_EMA_12),
                                    int(act_EMA_26),
                                    int(act_STD_rw),
                                    int(act_MXMN_rw)])
        return action_unscaled

    def construct_memory(self,window,action_unscaled,sharpe,daily_return,action):

        memory = pd.DataFrame(columns=["k_entry", "k_exit", "EMA_days_12",
                                       "EMA_days_26", "STD_rw", "MXMN_rw",
                                       "Sharpe", "Daily_returns_ROVI", "Position_ROVI"])
        memory.loc[window,"k_entry"] = action_unscaled[0]
        memory.loc[window, "k_exit"] = action_unscaled[1]
        memory.loc[window, "EMA_days_12"] = action_unscaled[2]
        memory.loc[window, "EMA_days_26"] = action_unscaled[3]
        memory.loc[window, "STD_rw"] = action_unscaled[4]
        memory.loc[window, "MXMN_rw"] = action_unscaled[5]
        memory.loc[window, "Sharpe"] = sharpe
        memory.loc[window, "Daily_returns_ROVI"] = daily_return
        memory.loc[window, "Position_ROVI"] = action

        return memory

    def step(self, actions):

        self.terminal = self.window >= len(self.dataframe) - 2
        action_unscaled = self.minmaxscaler_actions(actions)

        if self.terminal:
            print("Terminal")
            plt.plot(self.memory.Sharpe, 'b', label="Sharpes")
            plt.savefig(f'results/sharpe_{self.type}_{self.model_name}.png')
            plt.close()
            self.memory.to_csv(f"results/values_{self.type}_{self.model_name}.csv")
            return self.state, self.reward, self.terminal, {}

        else:

            self.window += 1
            data_win = self.dataframe.iloc[self.window - 100:self.window, :]
            self.data_window_RL = data_win.copy()
            self.data_window_NO_RL = data_win.copy()# I do this to avoid slicing problems
            # 0.85, 0.55, 3, 30,50,26

            strategy_RL = histogram_retracement(stock=stock,
                                             dataframe=self.data_window_RL,
                                             k_entry=action_unscaled[0],
                                             k_exit=action_unscaled[1],
                                             EMA_days_12=int(action_unscaled[2]),
                                             EMA_days_26=int(action_unscaled[3]),
                                             STD_rw=int(action_unscaled[4]),
                                             MXMN_rw=int(action_unscaled[5]))
            strategy_RL.signal_construction()
            sharpe_RL = self.getsharpe(self.data_window_RL)

            strategy_NO_RL = histogram_retracement(stock=stock,
                                            dataframe=self.data_window_NO_RL,
                                            k_entry=0.85,
                                            k_exit=0.55,
                                            EMA_days_12=3,
                                            EMA_days_26=30,
                                            STD_rw=50,
                                            MXMN_rw=26)
            strategy_NO_RL.signal_construction()
            sharpe_NO_RL = self.getsharpe(self.data_window_NO_RL)

            self.state = np.array([sharpe_RL, sharpe_NO_RL], dtype=np.float32)

            if sharpe_RL > sharpe_NO_RL:
                self.reward = 1
            if sharpe_RL < sharpe_NO_RL:
                self.reward = -1

            daily_returns = self.data_window_RL["Daily_returns_ROVI"][-1]
            action = self.data_window_RL["Position_ROVI"][-1]
            memory = self.construct_memory(self.window, action_unscaled, sharpe_RL, daily_returns, action)
            self.memory = pd.concat([self.memory, memory])

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.memory = pd.DataFrame(columns=["k_entry", "k_exit", "EMA_days_12",
                                            "EMA_days_26","STD_rw","MXMN_rw",
                                            "Sharpe","Daily_returns_ROVI", "Position_ROVI"])
        self.window = 100
        data_win = self.dataframe.iloc[self.window - 100:self.window, :]
        actions = self.action_space.sample()
        action_unscaled = self.minmaxscaler_actions(actions)
        # 0.85, 0.55, 3, 30,50,26
        self.data_window_RL = data_win.copy()
        self.data_window_NO_RL = data_win.copy()  # I do this to avoid slicing problems

        strategy_RL = histogram_retracement(stock=stock,
                                            dataframe=self.data_window_RL,
                                            k_entry=action_unscaled[0],
                                            k_exit=action_unscaled[1],
                                            EMA_days_12=int(action_unscaled[2]),
                                            EMA_days_26=int(action_unscaled[3]),
                                            STD_rw=int(action_unscaled[4]),
                                            MXMN_rw=int(action_unscaled[5]))
        strategy_RL.signal_construction()
        sharpe_RL = self.getsharpe(self.data_window_RL)

        strategy_NO_RL = histogram_retracement(stock=stock,
                                               dataframe=self.data_window_NO_RL,
                                               k_entry=0.85,
                                               k_exit=0.55,
                                               EMA_days_12=3,
                                               EMA_days_26=30,
                                               STD_rw=50,
                                               MXMN_rw=26)
        strategy_NO_RL.signal_construction()
        sharpe_NO_RL = self.getsharpe(self.data_window_NO_RL)
        daily_returns = self.data_window_RL["Daily_returns_ROVI"][-1]
        action = self.data_window_RL["Position_ROVI"][-1]
        #Memory of the most important variables
        memory = self.construct_memory(self.window, action_unscaled, sharpe_RL, daily_returns, action)
        self.memory = pd.concat([self.memory, memory])
        self.terminal = False

        # initiate state
        self.state = np.array([sharpe_RL, sharpe_NO_RL], dtype=np.float32)
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


