from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


import pandas as pd
import matplotlib.pyplot as plt

import joblib

iris = load_iris()

X = iris.data # shape (150, 4)
y = iris.target # shape (150,)

# print(iris.feature_names, iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# ============= Decision Tree

# before tuning with max depth
# model = DecisionTreeClassifier(random_state=42)

# tuning with max_depth = 5
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:5])
print(y_test[:5])

accuracy_test = accuracy_score(y_test, y_pred)

print(f"Accuracy Test: {accuracy_test}")

joblib.dump(model, "outputs/model.joblib")

# ============= K-NN

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"KNN Accuracy: {accuracy_knn}")

joblib.dump(model_knn, "outputs/model_knn.joblib")

# ============= Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.show()


