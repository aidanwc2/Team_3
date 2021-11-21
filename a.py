import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df_combined = pd.read_csv("CombinedDataMaleFinal.csv")
df_combined_a = df_combined[df_combined["ParentLocation"] == "Americas"]

df_combined_a = df_combined_a.dropna()

X = df_combined_a['Value'].values[:,np.newaxis]
y = df_combined_a['AvgCigarettePriceDollars'].values


model = LinearRegression()
model.fit(X,y)

plt.scatter(X, y)
plt.plot(X, model.predict(X),color='k')
plt.show()

print(model.coef_)
print(model.intercept_)