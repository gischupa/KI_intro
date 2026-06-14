import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

path = "abi.csv"
data = pd.read_csv(path, delimiter=';')

# Zielvariable
y = data["Bestanden"]

# Eingabemerkmale
X = data[["Fach", "Anwesend", "Übung"]]

# Kategoriale Variablen kodieren
X = pd.get_dummies(X)

# Zielvariable ebenfalls kodieren
y = y.map({"Nein": 0, "Ja": 1})

print(y)

# Baum trainieren
tr = tree.DecisionTreeClassifier(random_state=42)
tr.fit(X, y)

# Baum anzeigen
plt.figure(figsize=(10, 6))
tree.plot_tree(
    tr,
    feature_names=X.columns,
    class_names=["Nein", "Ja"],
    filled=True
)
plt.show()
