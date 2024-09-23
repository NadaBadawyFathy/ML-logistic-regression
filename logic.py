import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insurance_data.csv")
plt.scatter(df.age,df.bought_insurance,marker='+',color='red')

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
model = LogisticRegression()
model.fit(X_train, y_train)

print(model.predict(X_test))
print(model.predict_proba(X_test))
plt.show()