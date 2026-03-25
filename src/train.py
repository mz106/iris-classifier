import argparse

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


import pandas as pd
import matplotlib.pyplot as plt

import joblib

def parse_args():
	parser = argparse.ArgumentParser(description="Train Iris classifiers.")
	parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data used for testing.")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split.")
	return parser.parse_args()


def main():
	args = parse_args()

	iris = load_iris()

	X = iris.data
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=args.test_size,
		random_state=args.random_state,
	)

	model = DecisionTreeClassifier(random_state=args.random_state)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	accuracy_test = accuracy_score(y_test, y_pred)

	print(y_pred[:5])
	print(y_test[:5])
	print(f"Accuracy Test: {accuracy_test}")

	joblib.dump(model, "outputs/model.joblib")

	model_knn = KNeighborsClassifier(n_neighbors=5)
	model_knn.fit(X_train, y_train)
	y_pred_knn = model_knn.predict(X_test)
	accuracy_knn = accuracy_score(y_test, y_pred_knn)

	print(f"KNN Accuracy: {accuracy_knn}")

	joblib.dump(model_knn, "outputs/model_knn.joblib")

	cm = confusion_matrix(y_test, y_pred)
	print("Confusion Matrix:")
	print(cm)

	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
	disp.plot()
	plt.show()


if __name__ == "__main__":
	main()


