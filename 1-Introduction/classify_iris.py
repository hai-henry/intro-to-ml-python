"""
Application that Classifies Iris Species:
- Setosa
- Versicolor
- Virginica
This is a supervised learning problem, we know the correct species of iris and
the measurements and we want to predict the species which makes this a
classification problem. The different species are called classes.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt


def pair_plot_data(dataset, X_train, y_train):
    """
    Creates a pair scatter plot matrix for training data.

    Patamteters:
        dataset (dict): A dictionary containing feature data and target labels.
        X_train: Training features.
        y_train: Training target.

    Returns:
        plot: Pair scatter matrix
    """
    # Convert X_train to dataframe, with columns labeled with features
    iris_dataframe = pd.DataFrame(X_train, columns=dataset.feature_names)

    # Create scatter matrix from dataframe, coloring by y_train (output)
    pd.plotting.scatter_matrix(
        iris_dataframe,
        c=y_train,
        figsize=(15, 15),
        marker="o",
        s=60,
        hist_kwds={"bins": 20},
        alpha=0.8,
        cmap=mglearn.cm3,
    )
    plt.show()


def main():
    # Load iris dataset
    iris_dataset = load_iris()

    # Split data (NumPy array) into training and test set (75% train, 25% test)
    # Fix seed random_state = 0, ensure same split on runs
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset["data"], iris_dataset["target"], random_state=0
    )

    # Visualize training set
    pair_plot_data(iris_dataset, X_train, y_train)

    # Initialize KNN model and train (k = 1, single nearest neighbor)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # New data point we want to classify
    # [sepal length (cm)], sepal width (cm), petal length (cm), petal width (cm)
    X_new = np.array([[5, 2.9, 1, 0.2]])

    # Predict class with new data point
    prediction = knn.predict(X_new)
    print("Prediction: ", prediction)
    print("Prediction target: ", iris_dataset["target_names"][prediction])

    # Evaluating model for accuracy (correct species predicted) using test data
    y_predict = knn.predict(X_test)  # Makes predicitons with X_test features
    print("Test set prediction: \n", y_predict)  # Test predictions targets
    print("Test set targets: \n", y_test)  # Test actual targets

    # Print accracy two decimal points
    # knn.score = accuracy = (# of correct predictions)/(Total # of test samples)
    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

    # Calculate accuracy manually
    accuracy = np.mean(y_predict == y_test)
    print("Manual accuracy: {:.2f}".format(accuracy))


if __name__ == "__main__":
    main()
