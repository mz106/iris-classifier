import os
import subprocess
import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_train_script_generates_models():
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "src" / "train.py"
    model_path = project_root / "outputs" / "model.joblib"
    model_knn_path = project_root / "outputs" / "model_knn.joblib"
    test_size = 0.2
    random_state = 42

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--test-size",
            str(test_size),
            "--random-state",
            str(random_state),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Training script failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert model_path.exists(), "Decision Tree model file was not created."
    assert model_knn_path.exists(), "KNN model file was not created."

    model = joblib.load(model_path)
    model_knn = joblib.load(model_knn_path)

    assert hasattr(model, "predict")
    assert hasattr(model_knn, "predict")

    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=random_state,
    )

    decision_tree_accuracy = accuracy_score(y_test, model.predict(X_test))
    knn_accuracy = accuracy_score(y_test, model_knn.predict(X_test))

    assert decision_tree_accuracy > 0.9
    assert knn_accuracy > 0.9
