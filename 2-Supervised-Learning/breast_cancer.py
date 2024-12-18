"""
Not meant to be optimal, just function split be section in the book.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


def cancer_info():
    """
    For viewing keys, features, samples by class.
    """
    cancer = load_breast_cancer()

    print("Cancer keys: \n", cancer.keys())
    print("Cancer dataset shape: ", cancer.data.shape)

    zipped = zip(map(str, cancer.target_names), map(int, np.bincount(cancer.target)))
    print("Sample counts per class: \n", {n: v for n, v in zipped})

    print("Cancer dataset features: \n", cancer.feature_names)


def knn_accuracy():
    """
    For seeing KNeighborsClassifier model accuracy.
    """
    cancer = load_breast_cancer()

    # Split data set into training and test (75% train, 25% test)
    # Stratify to maintain proportions as dataset when splitting
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66
    )

    training_accuracy = []
    test_accuracy = []

    # k neighbors 1 to 10
    neighbors_settings = range(1, 11)

    for n_neighbors in neighbors_settings:
        # Build model and fit model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)

        # Record training accuracy
        training_accuracy.append(clf.score(X_train, y_train))

        # Record testing accuracy
        test_accuracy.append(clf.score(X_test, y_test))

    plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="Testing accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()


def linear_logistic():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42
    )

    logreg = LogisticRegression().fit(X_train, y_train)
    print(f"Training set score: {logreg.score(X_train,y_train):.3f}")
    print(f"Test set score: {logreg.score(X_test,y_test):.3f}")

    logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
    print(f"Training set score: {logreg100.score(X_train,y_train):.3f}")
    print(f"Test set score: {logreg100.score(X_test,y_test):.3f}")

    logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print(f"Training set score: {logreg001.score(X_train,y_train):.3f}")
    print(f"Test set score: {logreg001.score(X_test,y_test):.3f}")

    plt.plot(logreg.coef_.T, "o", label="C=1")
    plt.plot(logreg100.coef_.T, "^", label="C=100")
    plt.plot(logreg001.coef_.T, "v", label="C=0.001")
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient maginitude")
    plt.legend()
    plt.show()


def decision_tree():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42
    )

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)

    print(f"Training set score: {tree.score(X_train,y_train):.3f}")
    print(f"Test set score: {tree.score(X_test,y_test):.3f}")

    # Apply pre-pruning (stop building tree at depth)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X_train, y_train)

    print(f"Pre-pruned Training set score: {tree.score(X_train,y_train):.3f}")
    print(f"Pre-pruned Test set score: {tree.score(X_test,y_test):.3f}")

    export_graphviz(
        tree,
        out_file="tree.dot",
        class_names=["Malignant", "benign"],
        feature_names=cancer.feature_names,
        impurity=False,
        filled=True,
    )

    plot_feature_importances(cancer, tree)


def plot_feature_importances(dataset, model):
    n_features = dataset.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()


def main():
    decision_tree()


if __name__ == "__main__":
    main()
