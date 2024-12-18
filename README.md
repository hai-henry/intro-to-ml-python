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
  - High complexity
- **Underfitting:** Occurs when the model is too simple. Performs poorly on training set, does not capture the underlying patterns and relationships in data. 
  - Low complexity
	- Can happen when the model is generalizing too much (oversimplifying data)
		- eg. Everybody who owns a phone has an iPhone

<div align="center">	
	<img src="https://github.com/user-attachments/assets/41702d51-08b5-4726-bc29-0247403e56ed" alt="Model complexity" width="600px">
</div>

### Linear Models
Linear models make predictions using a linear function of input features. Widely used in practice.
- For datasets with many features, linear models can be very powerful
	- if you have more features than data points
#### Linear models for regression (general formula)
$$\hat{y}=w[0]*x[0]+w[1]*x[1]+...+w[p]*x[p]+b$$
- $\hat{y}$ is the prediction the model makes
- $x[0]$ to $x[p]$ are features
- $w$ and $b$ are learned parameters from model
 **Linear regression for single feature**:
$$\hat{y} = w[0] *x[0] +b$$
#### Ridge Regression
Also linear model for regression, formula for prediction is the same for ordinary least squares.
- Coefficients $w$ chosen to predict well on training data but also fit constraints.
	- Want $w$ to be close to zero
- More restricted, less likely to overfit
- Features should have little effect on outcome as possible, while predicting well. This constraint is called ***regularization***, meaning explicitly restricting model to avoid overfitting.
	- In ridge regression, particular kind used is **L2 regularization**
- increasing alpha = more restrict
- decreasing alpha = less restrict
- With more data, regularization becomes less important
	- Given enough data, ridge and linear regression will have same performance
#### Lasso
Alternative to ridge, for regularizing linear regression. Restricts coefficient closer to zero but through **L1 regularization**.
- When using L1 Regularization, some coefficients are exactly zero
	- Means some features ignored by model
		- makes model easier to interpret
		- can real most important features of model
- Ridge usually first choice between the two
- If you have large amount of features but only expect a few to be important, Lasso might be better
##### Example: Lasso
- Training set score: 0.29
- Test set score: 0.21
- Number of features used: 4 
Bad performance on both training and test. We are underfitting, lasso also only used 4 of 105 features. Alpha is at alpha = 1.0, decrease alpha to reduce underfitting and increase max_iter.
 - Training set score: 0.90
- Test set score: 0.77
- Number of features used: 33
Lower alpha allowed to fit a more complex model, performance is better. If alpha goes too low, we lose regularization.
#### Linear models for classification
Extensively used for classification. **Binary classification** formula:
$$\hat{y}=w[0]*x[0]+w[1]*x[1]+...+w[p]*x[p]+b > 0$$
**Common models:**
- Logistic regression
- Linear support vector machines (Linear SVMs)
Strength of regularization is called C.
- Higher C corresponds to less regularization
- low value C, algo tries to adjust majority of data points
- higher value of C, importance that each datapoint be classified correctly
#### Linear models for multiclass classification
- For thousands or millions of datasets use solver='sag' option in LogisticRegression and ridge, faster than default
### Decision Trees
- Widely used for classification and regression tasks
- Essentially learn if/else into a decision
- Preventing overfitting:
	- Pre-pruning
	- Pruning
- *Feature importance*, rates how important each feature is for decision tree
