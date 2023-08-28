import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

example = datasets.load_breast_cancer()
x, y = example.data, example.target

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1234)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
  #we need to define the learning rate and the number of iterations
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    #we need to define the fit function
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            # we need to calculate the gradients
            # dw is the gradient of the weights
            # db is the gradient of the bias
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    # we need to define the predict function which will return the class predictions
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    
# we need to define the accuracy function which will return the accuracy of the model
def accuracy (y_pred, y_test):
        return np.sum(y_pred==y_test)/len(y_test)
    
iterations=[]
accuracy=[]

regressor = LogisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(xtrain, ytrain)
predictions = regressor.predict(xtest)
iterations.append(1000)

print("Accuracy score of the model is: ", accuracy(predictions, ytest)*100, "%")
accuracy.append(accuracy(predictions, ytest))

print(regressor.weights)
print(regressor.bias)

plt.scatter(iterations,accuracy,color="o")
plt.plot(iterations,accuracy)
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")


