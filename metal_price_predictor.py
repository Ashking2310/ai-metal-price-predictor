import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample historical LME data
data = {
    "Day": [1,2,3,4,5,6,7,8,9,10],
    "Aluminium_LME": [2200,2215,2230,2220,2245,2260,2275,2290,2300,2315],
    "Copper_LME": [8500,8520,8550,8530,8570,8600,8625,8650,8680,8700]
}

df = pd.DataFrame(data)

# Prepare model for Aluminium
X = df[["Day"]]
y_al = df["Aluminium_LME"]

model_al = LinearRegression()
model_al.fit(X, y_al)

# Prepare model for Copper
y_cu = df["Copper_LME"]

model_cu = LinearRegression()
model_cu.fit(X, y_cu)

# Predict future price
future_day = np.array([[12]])

al_prediction = model_al.predict(future_day)
cu_prediction = model_cu.predict(future_day)

print("Predicted Aluminium LME price:", al_prediction[0])
print("Predicted Copper LME price:", cu_prediction[0])
