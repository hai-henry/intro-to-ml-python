import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_plot(X, y):
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

    # New data points are stars
    mglearn.plots.plot_knn_classification(n_neighbors=3)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()


def dec_boundaries_plot(X, y):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title(f"{n_neighbors} neighbors")
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")

    axes[0].legend(loc=3)
    plt.show()


def main():
    X, y = mglearn.datasets.make_forge()

    # Split data set into training and test (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Build and fit model
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    # Make prediction on test data
    prediction = clf.predict(X_test)
    print("Test predition: \n", prediction)

    # Evaluate model accuracy
    accuracy = clf.score(X_test, y_test)
    print(f"Test set accuracy:  {accuracy:.2f}")

    dec_boundaries_plot(X, y)


if __name__ == "__main__":
    main()
