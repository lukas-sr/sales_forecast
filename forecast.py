import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Dataframe
df = pd.read_csv("advertising.csv")

# print(df.info())

# Data visualizer to correlation between the variables in our dataframe
# sns.pairplot(df)
# plt.show()

# Heatmap
# sns.heatmap(df.corr(), cmap = "Wistia", annot = True)

# Training AI
x = df.drop("Vendas", axis = 1)
y = df["Vendas"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)

# Testing both models
test_pred_lin = lin_reg.predict(x_test)
test_pred_rf = rf_reg.predict(x_test)

r2_lin = metrics.r2_score(y_test, test_pred_lin)
mse_lin = metrics.mean_squared_error(y_test, test_pred_lin)

print(f"R² Regressão Linear: {r2_lin}")
print(f"MSE Regressão Linear: {mse_lin}")

r2_rf= metrics.r2_score(y_test, test_pred_rf)
mse_rf = metrics.mean_squared_error(y_test, test_pred_rf)

print(f"R² Random Forest: {r2_rf}")
print(f"MSE Random Forest: {mse_rf}")
