# intro-to-ml-python
Introduction to Machine Learning with Python, Andreas C. Muller, Sarah Guido 

## 1. Introduction
In essence, Machine Learning is about seeing what we can know from data. It is
a field that interconnects statistics, artificial intelligence, and computer 
science and is also known as predictive analytics or statistical learning.
### Supervised Learning
ML algorithm that learns from input/output pairs. User provides inputs and the desired outputs.
### Unsupervised Learning
Typically input is given, and output is not. eg. Categorizing customers into groups with similar preferences
- Given a set of customer records, you might want to identify customer similarities and categorize them. Could be 
"athletes", "non-athletes", "parents". Because you do not know these customers beforehand, you do not know where
to categorize them, you have no known outputs.
### K-Nearest Neighbors
- Given a new data point
- Find the nearest point (neighbor)
	- k = 1, finding the **single** nearest neighbor
	- e.g k = 3 or 5, finding the nearest neighbors
- Assign label to the new data point
- e.g scatter plot with classes = red, blue, green k = 5
	- New point given
	- Nearest neighbors: 2 reds, 3 blues
	- New point classifies as blue
## 2. Supervised Learning
Two major types of supervised machine learning problems:
- Classification
- Regression
### Classification and Regression
The goal of **classification** is to predict a *class label*,in [[#1. Introduction|chapter 1]], we created an application that classified irises into one of the three species. Classification can also be separated into:
- Binary classification
- Multiclass classification
The goal of **regression** is to predict a continuous number. eg. Predicting a person's annual income based off their education, age, location. Typically regression has continuity in the output unlike classification.
### Generalization, Overfitting, and Underfitting
- **Generalization:** Model is able to make accurate predictions on unseen data
	- Typically want model to generalize as accurate as possible
- **Overfitting:** Occurs when the model is fit too closely to the training set and works well with the training set but not able to generalize to new data.
	- Can happen when building a model that is too complex for the amount of data we have
- **Underfitting:** Occurs when the model is too simple. Performs poorly on training set, does not capture the underlying patterns and relationships in data. 
	- Can happen when the model is generalizing too much (oversimplifying data)
		- eg. Everybody who owns a phone has an iPhone

<div align="center">	
	<img src="https://github.com/user-attachments/assets/41702d51-08b5-4726-bc29-0247403e56ed" alt="Model complexity" width="600px">
</div>
