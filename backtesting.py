import pandas as pd

from main import dataframe, stock
from histogram_retracement import histogram_retracement
import itertools
# Define all posible combinations for the backtesting

aux = dataframe.loc[(dataframe.index >= "2007-12-06") & (dataframe.index < "2018-12-31")].copy()

params = {"k_entry": [0.55, 0.75, 0.85, 0.95], "k_exit": [0.15, 0.55, 0.75, 0.95],
          "EMA_days_12": [3, 5, 10, 12], "EMA_days_26": [20, 26, 30, 50],
          "STD_rollingwindow": [10, 20, 30, 50], "MAXMIN_rollingwindow": [10, 26, 30, 50]}
keys = params.keys()
values = (params[key] for key in keys)
comb = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

returns_, Avg_Sharp_mem = [], []

for z in range(len(comb)):
    # apply trading strategy
    k_entry = comb[z]["k_entry"]  # percentage of peaks and througs
    k_exit = comb[z][
        "k_exit"]  # solo funciona entre 0.23 y 0.53 cuando vale 0.75, esto es porque no hay signal en ese año
    EMA_days_12 = comb[z]["EMA_days_12"]
    EMA_days_26 = comb[z]["EMA_days_26"]
    STD_rw = comb[z]["STD_rollingwindow"]
    MXMN_rw = comb[z]["MAXMIN_rollingwindow"]
    strategy = histogram_retracement(stock=stock, dataframe=aux, k_entry=k_entry, k_exit=k_exit,
                                     EMA_days_12=EMA_days_12, EMA_days_26=EMA_days_26, STD_rw=STD_rw,
                                     MXMN_rw=MXMN_rw)
    strategy.signal_construction()
    ret, Avg_sharpe = strategy.get_returns()
    returns_.append(ret)
    Avg_Sharp_mem.append(Avg_sharpe)
    print(f"Iteración {z} de {len(comb) - 1}", end="\r")

print(f"""El maximo sharp medio conjunto es: {max(Avg_Sharp_mem)} 
    se da para el indice: {Avg_Sharp_mem.index(max(Avg_Sharp_mem))}
    y para la combinacion: {comb[Avg_Sharp_mem.index(max(Avg_Sharp_mem))]}
    La tabla de retornos es: """)  # returns_
print("\n")

for z in range(len(comb)):
    comb[z]["Sharpe"]=Avg_Sharp_mem[z]
result = pd.DataFrame.from_records(comb)
result.to_csv("Chooseapathyourself")


"""Resultado: El maximo sharp medio conjunto es: 3.180057128283203 
    se da para el indice: 2349
    y para la combinacion: {'k_entry': 0.85, 'k_exit': 0.55, 'EMA_days_12': 3, 'EMA_days_26': 30, 'STD_rollingwindow': 50, 'MAXMIN_rollingwindow': 26}
    La tabla de retornos es: """


