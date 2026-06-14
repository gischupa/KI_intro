import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

path = "abi.csv"
data = pd.read_csv(path, delimiter=';')

col_name = 'Bestanden'
abi_right = data[col_name]
abi_left = data[ ["Fach", "Anwesend", "Übung"]  ]


tr = tree.DecisionTreeClassifier()
tr.fit(abi_left, abi_right)
tree.plot_tree(tr)
