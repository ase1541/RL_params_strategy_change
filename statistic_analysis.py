import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from main import data, stock

data.drop(columns=["Currency"], inplace = True)


##ANALISIS DESCRIPTIVO
data["Daily_returns"]=data["Close"].pct_change()
data.fillna(0)
data = data.iloc[1:]

plt.figure(figsize=(15,8))
sns.set(color_codes = True)
ax = sns.distplot(data["Daily_returns"], bins=100,kde=False,fit=stats.norm,color="red")
(mu,sigma)=stats.norm.fit(data["Daily_returns"])
#Print de la funcion de desnsidad de probabilidad de la distribución de los retornos diarios
plt.title(f"Historical Daily Return Distribution {stock}")
plt.ylabel("Frequence")
plt.legend([f"Normal Distribution. fit mu:{round(mu,6)}, sigma:{round(sigma,3)} {stock}",f"Daily returns distribution {stock}"])
#plt.show()


##PRINT DE DISTINTAS MÉTRICAS IMPORTANTES
years=data["Daily_returns"].count() / 252 #We assume 252 traiding days
CAGR=(data["Close"].iloc[-1] / data["Close"].iloc[0]) ** (1 / years) -1
B_y_H=(100*((data["Close"].iloc[-1] - data["Close"].iloc[0])/ data["Close"].iloc[0]))
i=51
print(i*"=")
print(f' > Tasa de crecimiento anual compuesto: {round(100*CAGR,2)} %')
print(f' > Buy & Hold: {round(B_y_H,2)} %')

Maximo_Anterior = data["Daily_returns"].cummax()
drawdowns= 100 * ((data["Daily_returns"] - Maximo_Anterior)/Maximo_Anterior)
DD = pd.DataFrame({"Close": data["Daily_returns"], "Previous Peak":Maximo_Anterior, "Drawdonwns": drawdowns})

print(f" > Máximo Drawdown Historico: {round(np.min(DD['Drawdonwns']),2)} %")
print(f" > Media diaria: {round(100*data['Daily_returns'].mean(),5)} %")
print(f" > Desviación típica diaria: {round(100*data['Daily_returns'].std(),2)} %")
print(f" > Máxima pérdida diaria: {round(100*data['Daily_returns'].min(),2)} %")
print(f" > Máxima beneficio diario: {round(100*data['Daily_returns'].max(),2)} %")
print(f" > Días analizados: {round(data['Daily_returns'].count(),2)}")
print(i*"=")
print(f' > Coeficiente de asimetría: {round(data["Daily_returns"].skew(),4)}')
print(f' > Curtosis: {round(data["Daily_returns"].kurt(),4)}') #Tells how tails deffer from those of a normal distribution (3)
print(i*"=")
print(f" > Value at Risk Modelo Gauss NC-95%: {round(100* norm.ppf(0.05,mu,sigma),3)} %")
print(f" > Value at Risk Modelo Gauss NC-99%: {round(100* norm.ppf(0.01,mu,sigma),3)} %")
print(f" > Value at Risk Modelo Histórico NC-95%: {round(100* np.percentile(data['Daily_returns'],5),3)} %")
print(f" > Value at Risk Modelo Histórico NC-99%: {round(100* np.percentile(data['Daily_returns'],1),3)} %")
print(i*"=")


##ANALISIS DE LA VOLATILIDAD
data["Volatilidad_His_20dias"]=1000*data["Daily_returns"].rolling(20).std()
data["Volatilidad_His_20dias_Anulizada"]=data["Volatilidad_His_20dias"]
data["SMA_Volatilidad_Anualizada"]=data["Volatilidad_His_20dias"].rolling(126).mean()

fig, ax1 = plt.subplots(figsize=(15,8))
ax2 = ax1.twinx()
volatilityLine=ax1.plot(data["Volatilidad_His_20dias_Anulizada"],'blue',label="Vol 20 days Annualized")
smaLine=ax1.plot(data["SMA_Volatilidad_Anualizada"],'red',label="SMA 126 days Annualized")
close_price=ax1.plot(data["Close"],'black',label="Closing price")


plt.title(f"Historical evolution of price and volatility for {stock}", fontsize=16)

ax1.set_xlabel("Date")
ax1.set_ylabel("Annualized Volatility", color="black")
ax2.set_ylabel("Closing price", color="black")

ax1.legend()
ax2.legend()
#plt.show()


##ANALISIS DE DICKEY-FULLER
adf =adfuller(data["Close"], maxlag=1)
print(f"El T-test es: {adf[0]}")
print(f"El P-value es: {adf[1]}")
print(f"Valores criticos: {adf[4]}")