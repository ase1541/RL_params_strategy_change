# RL_trading_params_change

The very AIM of this project is to build a RL Agent able to find the optimal parameters for a trading strategy called Histogram Retracement.
Therefore, the Agent will select the parameters more suitable to maximize sharp ratio of each transaction. There are several scripts in this project. There are two big groups of scripts:

<ins>**Scripts related to the traiding strategy**<ins>
  
1/ **main.py**: It is the general code where the general calls to the other scripts are done. Here we obtain the dataset for the stock that we want to study
through the library invest.py.

2/**histogram_retracement.py**: It is nothing but the trading strategy. In order to make things easier, it is implemented as a class where you have initialize 
the following attributes:
  - **stock** = stock name used for the library to obtain the dataset
  - **dataframe** = dataframe where the stock prices are stored
  - **k_entry and k_exit** are the parameters that manage to take the decisions of when to generate entry signals and exit signals with respect the Histogram 
  retracement graphic (It is continuously crossing zero, reaching a maximum, and then doing just the opposite)
  - **EMA_days_12 and EMA_days_26** are both the days needed to construct the moving average that goes fast and the one that goes slow for the MACD part of the
  strategy.
  - **STD_rw and MXMN_rw** are the rolling windows for both the standard deviation and the maximums that describe the conditions for creating the entry signals.

Once we have described the attributes of the class, we describe the methods it has:
  - **count_trades**: It counts the number of trades done and also provides a dictionary that contains start, end, Number of days, return and sharp ratio of each
  trading done throughout time
  - **signal_construction**: creates the different columns containing the strategy, so that the initial dataframe that only contains a column with the closing price
  observation, has additionally the Histogram retracement, selling (short) signals, buying (long) signals, Position and daily return columns.
  - **sell_buy_function**: function needed by the signal_construction function to create the selling and buying signals
  - **get_max_min**: function that obtains the max and min values for a given block of the Histogram retracement above or under zero. It is currently not used,
  because it would need us to know the future values of the Histogram Retracement, which in practice we don't.
  - **plot_signals**: Returns 2 figures, one with the stock prices and each selling and buying signal marked, and the same but for the histogram retracement graphic.
  - **get_returns**: Creates a pd.DataFrame that contains sharp, return and volatility both for short and long, annually. The last row of the frame is the average of
  these values.
  
3/**backtesting.py**: It is mandatory to perform a study of the best suiting parameters so we can simplify the labor later performed by the RL agent, hence a 
backtesting is performed in order to retrieve the parameters that maximize the return of the strategy. We select discrete values for each of the attributes described 
previously and generate all the possible combinations of these ones. After trying with 5 possible options for each parameter, we end up with 4096 possible combinations
  
4/**statistic_analysis.py**: In this script an statistical analysis of the time series is performed.

<ins>**Scripts related to RL**<ins>
  
5/**Rl_algo_trading**: In this script the environment class for the RL model is defined, as well as the different agents and some functions that make an easier and smoother funtioning of the next script.

6/**train_RL.py**: Here happens the training and validation of the different RL models. THe result obtained is A2C as the best model and therefore, only this one is trained long enough to obtained the final results.

7/**performance.py**: in this script lies the comparison between the RL algorithm performance and the strategy with fixed parameters.

  
  
Trained models have been stored in the models folder, and several results are stored in the results folder as well. This code is the result of a final thesis and was develop from february 2022 to june 2022.
