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
