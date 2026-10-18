import yfinance as yf

data = yf.download("USDMYR=X", start="2015-01-01")
data.to_csv("ringgit_history.csv")
print(data.tail())
print("Rows:", len(data))
