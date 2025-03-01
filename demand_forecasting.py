# %% Load Data
import pandas as pd
import psycopg2
from pyspark.sql import SparkSession

df = pd.read_csv("/Users/yaldaradan/Downloads/stores_sales_forecasting.csv", encoding='latin-1')

#print(df.head())


spark = SparkSession.builder \
    .appName("DemandForecasting") \
    .getOrCreate()

sales_data = spark.createDataFrame(df)
#sales_data.show()

daily_sales = sales_data.groupBy("Order Date", "Product ID").sum("Quantity")
#daily_sales.show()
# %%
import os
os.system("airflow dags trigger demand_forecasting")
# %%
df["Order Date"] = pd.to_datetime(df["Order Date"])
df = df.sort_values(by="Order Date")

df["Day of the week"] = df["Order Date"].dt.dayofweek
df["Month"] = df["Order Date"].dt.month
df["Year"] = df["Order Date"].dt.year
df["Week of year"] = df["Order Date"].dt.isocalendar().week
df["Prevoius day sales"] = df.groupby("Product ID")["Quantity"].shift(1).fillna(0)
df["Prevoius week sales"] = df.groupby("Product ID")["Quantity"].shift(7).fillna(0)


print(df.dtypes)


# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

df_new = df[["Row ID", "Order ID", "Product ID", "Order Date", "Day of the week", "Month", "Year", "Week of year", "Sales", "Prevoius day sales", "Prevoius week sales", "Quantity"]]
df_new["Product ID"] = df_new["Product ID"].astype("category").cat.codes.fillna(-1)
df_new["Order ID"] = df_new["Order ID"].astype("category").cat.codes.fillna(-1)
print(df_new.dtypes)

# %%
x = df_new[["Product ID", "Day of the week", "Month", "Year", "Week of year", "Sales", "Prevoius day sales", "Prevoius week sales"]]
y = df_new["Quantity"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.1, 0.01, 0.001]}
grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_['n_estimators'])

# %%
n_estimators = grid_search.best_params_['n_estimators']
model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(predictions)

# %%
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, predictions, squared=False)
print("XGBoost RMSE:", rmse)
# %%
df = pd.read_excel("/Users/yaldaradan/Downloads/customer.xlsx")
print(df.head())
print(df["Quantity"].min())
# %%
print(df["Quantity"].min())
print(df["Quantity"].describe())
print("Negative Transactions:", (df["Quantity"] < 0).sum())
print("Positive Transactions:", (df["Quantity"] > 0).sum())

print(df["StockCode"].value_counts().max())
# %%
df = df[df["Quantity"] > 0] 
df_lstm = df[df["StockCode"] == "85123A"]
df_lstm.set_index("InvoiceDate", inplace=True)
sales_data_lstm = df_lstm["Quantity"].values.reshape(-1, 1)
print(sales_data_lstm)

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
sales_data_lstm = scaler.fit_transform(sales_data_lstm)
print(sales_data_lstm)
print(sales_data_lstm.shape)
# %%
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

generator = TimeseriesGenerator(sales_data_lstm, sales_data_lstm, length=30, batch_size=1)
x, y = generator[0]
print(x)
print(y)


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([LSTM(50, return_sequences=True, input_shape=(30, 1)), LSTM(50), Dense(1)])

model.compile(optimizer='adam', loss='mse')
model.summary()

# %%
model.fit(generator, epochs=20)
# %%
import numpy as np
last_sequence = sales_data_lstm[-30:]   
last_sequence = np.expand_dims(last_sequence, axis=0)

# %%
forecast = model.predict(last_sequence)
forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
print(forecast[0][0])
# %%
future_forecast = []
sales_data_temp = sales_data_lstm.copy()
for i in range(7):
    last_sequence = sales_data_temp[-30:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    forecast = model.predict(last_sequence)
    future_forecast.append(forecast[0][0])
    sales_data_temp = np.append(sales_data_temp, forecast).reshape(-1, 1)

future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
print(future_forecast)



# %%
