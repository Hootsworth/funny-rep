# train.py
# Basic Random Forest example using sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    print("Loading dataset...")
    data = load_iris()

    X = data.data
    y = data.target

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Predicting...")
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
