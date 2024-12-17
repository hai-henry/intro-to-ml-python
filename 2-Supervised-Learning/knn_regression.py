import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


def knn_regress_plot(X, y):
    mglearn.plots.plot_knn_regression(n_neighbors=3)

    plt.plot(X, y, "o")
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()


def analyze_regressor(X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Create 1000 datapoints, spaced between -3 and 3 evenly
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)

    for n_neighbors, ax in zip([1, 3, 9], axes):
        # Build and fit model
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)

        # Make predictions
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, "^", c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, "v", c=mglearn.cm2(1), markersize=8)

        ax.set_title(
            f"{n_neighbors} neighbors(s)\n"
            f"Train score: {reg.score(X_train, y_train):.2f} "
            f"Test score: {reg.score(X_test,y_test):.2f}"
        )
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")

    axes[0].legend(
        ["Model predicitons", "Training data/target", "Test data/target"], loc="best"
    )
    plt.show()


def main():
    X, y = mglearn.datasets.make_wave(n_samples=40)

    # Split data set into training and test (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Build model and fit
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)

    # Test predictions
    print("Test set predicitons: \n", reg.predict(X_test))

    # Evaluate model using R2 score (coefficient of determination)
    # Score between 0 to 1
    print(f"Test set R^2: {reg.score(X_test,y_test):.2f}")

    analyze_regressor(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
