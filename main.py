import investpy
import datetime as dt
from histogram_retracement import *
# Define parameters for the investpy API
first_date = '06/12/2007'
now = dt.datetime.today()
today = now.strftime('%d/%m/%Y')
stock = "ROVI"

# get stock data
data = investpy.stocks.get_stock_historical_data(stock,
                                                 country='spain',
                                                 from_date=first_date,
                                                 to_date=today,
                                                 interval='Daily')
dataframe = data.copy()
dataframe.drop(columns=["Currency","Open","High","Low","Volume"], inplace = True)
dataframe.rename(columns={"Close": stock}, inplace=True)





