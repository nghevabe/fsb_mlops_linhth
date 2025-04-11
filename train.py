from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("classification_experiment")

params_list = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 300, "max_depth": 20}
]

best_acc = 0

for params in params_list:
    with mlflow.start_run():
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        if acc > best_acc:
            best_acc = acc
            mlflow.sklearn.log_model(clf, "model", registered_model_name="BestClassifierModel")