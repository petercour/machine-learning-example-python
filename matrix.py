from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np

# url with dataset
#url = "https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv"

# load the CSV file as a numpy matrix
dataset = np.loadtxt("./pima-indians-diabetes.csv", delimiter=",")

# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]

# Initialize algorithm
model = LogisticRegression()
model.fit(X, y)

print('MODEL')
print(model)

# make predictions
expected = y
predicted = model.predict(X)

# summarize the fit of the model
print('RESULT')
print(metrics.classification_report(expected, predicted))
print('CONFUSION MATRIX')
print(metrics.confusion_matrix(expected, predicted))
