'''
Application that Classifies Iris Species:
- Setosa
- Versicolor
- Virginica
This is a supervised learning problem, we know the correct species of iris and
the measurements and we want to predict the species which makes this a
classification problem. The different species are called classes.
'''
from sklearn.datasets import load_iris

iris_dataset = load_iris()
