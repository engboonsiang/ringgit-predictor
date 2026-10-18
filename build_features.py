import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("ringgit_history.csv", skiprows=2, names=["Date", "Close", "High", "Low", "Open", "Volume"])
df = df[["Date", "Close"]].dropna()
df["Close"] = df["Close"].astype(float)

df["lag1"] = df["Close"].shift(1)
df["lag2"] = df["Close"].shift(2)
df["lag3"] = df["Close"].shift(3)
df["target"] = df["Close"]
df = df.dropna()

train = df.iloc[:-60]
test = df.iloc[-60:]

print("Train rows:", len(train))
print("Test rows:", len(test))

feature_cols = ["lag1", "lag2", "lag3"]
X_train = train[feature_cols]
y_train = train["target"]
X_test = test[feature_cols]
y_test = test["target"]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

plt.figure(figsize=(10, 5))
plt.plot(test["Date"].values, y_test.values, label="Real price")
plt.plot(test["Date"].values, predictions, label="Predicted price")
plt.xticks(rotation=45)
plt.legend()
plt.title("Ringgit (MYR/USD): Real vs Predicted")
plt.tight_layout()
plt.savefig("prediction_plot.png")
print("Saved plot: prediction_plot.png")
