"""
Application that Classifies Iris Species:
- Setosa
- Versicolor
- Virginica
This is a supervised learning problem, we know the correct species of iris and
the measurements and we want to predict the species which makes this a
classification problem. The different species are called classes.
"""

from sklearn.datasets import load_iris  # Import sklearn dataset
from sklearn.model_selection import train_test_split  # Dataset split function
import pandas as pd
import mglearn
import matplotlib.pyplot as plt


def print_data_info(dataset):
    # Keys of the dataset
    # print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

    # Take the key (target_names) and print the targets (setosa, versicolor,
    # virginica)
    # print("Target names: {}".format(iris_dataset['target_names']))

    # Print features (sepal (len & wid, petal (len & wid)))
    # print("Feature names: {}".format(iris_dataset['feature_names']))

    # Print features, target
    for features, target in zip(dataset["data"], dataset["target"]):
        print(f" {features}, {target}")


def pair_plot_data(dataset, X_train, y_train):
    # Convert X_train to dataframe
    # Columns are labeled with features
    iris_dataframe = pd.DataFrame(X_train, columns=dataset.feature_names)

    # Create scatter matrix from dataframe, color by y_train (output)
    pd.plotting.scatter_matrix(
        iris_dataframe,
        c=y_train,
        figsize=(15, 15),
        marker="o",
        hist_kwds={"bins": 20},
        s=60,
        alpha=0.8,
        cmap=mglearn.cm3,
    )
    plt.show()


def main():
    iris_dataset = load_iris()  # Load the dataset to variable

    # Split data into training and test set (75%, 25%)
    # Fix seed random_state = 0, so it does not shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset["data"], iris_dataset["target"], random_state=0
    )

    pair_plot_data(iris_dataset, X_train, y_train)


if __name__ == "__main__":
    main()
