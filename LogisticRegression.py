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

             # Check if the current iteration is in the desired iterations list
            if _ + 1 in list_iterations:
                accuracy_score = accuracy(self.predict(xtest), ytest)
                accuracy_scores.append(accuracy_score)


    # we need to define the predict function which will return the class predictions
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    
# we need to define the accuracy function which will return the accuracy of the model
def accuracy (y_pred, y_test):
        return np.sum(y_pred==y_test)/len(y_test)
    
list_iterations = [100, 300, 500, 700, 1000]
accuracy_scores=[]

regressor = LogisticRegression(lr=0.0001, n_iters=1000)
regressor.fit(xtrain, ytrain)
predictions = regressor.predict(xtest)


print("Accuracy score of the model is: ", accuracy(predictions, ytest)*100, "%")
accuracy_scores.append(accuracy(predictions, ytest))

print(regressor.weights)

print(list_iterations)
print(accuracy_scores)

accuracy_scores.pop()
plt.scatter(list_iterations, accuracy_scores, s=50, c='blue', marker='o')
plt.plot(list_iterations,accuracy_scores)
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.show()


