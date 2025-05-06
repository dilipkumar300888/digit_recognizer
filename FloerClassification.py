from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data,columns=data.fearure_names)
df['species'] = data.target


from sklearn.model_selection import train_test_split

X = df[data.feature_names]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df, hue='species')
plt.show()