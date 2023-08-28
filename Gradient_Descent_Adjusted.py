# use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)
import matplotlib.pyplot as plt
import numpy  # numpy is used to make some operations with arrays more easily

__errors__ = []  # global variable to store the errors/loss for visualisation


def h(params, sample):
    """
    Returns the estimated value for a given input sample and a given set of parameters

    Args:
            params (lst) a list containing the corresponding parameter for each element x of the sample
            sample (lst) a list containing the input sample

    Returns:
            (float) the estimated value of the sample using the given parameters
    """
    hyp = 0
    for i in range(len(sample)):
        hyp += params[i]*sample[i]
    return hyp


def show_errors(params, samples, y):
    """Appends the errors/loss that are generated by the estimated values of h and the real value y

    Args:
            params (lst) a list containing the corresponding parameter for each element x of the sample
            samples (lst) a 2 dimensional list containing the input samples 
            y (lst) a list containing the corresponding real result for each sample

    """
    global __errors__
    error_acum = 0
# print("transposed samples")
# print(samples)
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        print("hyp  %f  y %f " % (hyp,  y[i]))
        error = hyp-y[i]
        # this error is the original cost function, (the one used to make updates in GD is the derivated version of this formula)
        error_acum = +error**2
    mean_error_param = error_acum/len(samples)
    __errors__.append(mean_error_param)


def GD(params, samples, y, alfa):
    """Gradient Descent algorithm 
    Args:
            params (lst) a list containing the corresponding parameter for each element x of the sample
            samples (lst) a 2 dimensional list containing the input samples 
            y (lst) a list containing the corresponding real result for each sample
            alfa(float) the learning rate
    Returns:
            temp(lst) a list with the new values for the parameters after 1 run of the sample set
    """
    temp = list(params)
    for j in range(len(params)):
        acum = 0
        for i in range(len(samples)):
            error = h(params, samples[i]) - y[i]
            # Sumatory part of the Gradient Descent formula for linear Regression.
            acum = acum + error*samples[i][j]
        # Subtraction of original parameter value with learning rate included.
        temp[j] = params[j] - alfa*(1/len(samples))*acum
    return temp


"""
Normalizes sample values so that gradient descent can converge
  Args:
    samples (lst) a list containing the corresponding parameter for each element x of the sample
  Returns:
    samples(lst) a list with the normalized version of the original samples
"""


def scaling(samples):
    # This code takes each sample and subtracts the mean and then divides by the max value. This is mean scaling.
    # We do this so that the data is between -1 and 1
    samples = numpy.asarray(samples).T.tolist()
    for i in range(1, len(samples)):
        acum = 0
        for j in range(len(samples[i])):
            acum += samples[i][j]
        avg = acum / (len(samples[i]))
        max_val = max(samples[i])
        # print("avg %f" % avg)
        # print(max_val)
        for j in range(len(samples[i])):
            # print(samples[i][j])
            samples[i][j] = (samples[i][j] - avg) / max_val  # Mean scaling
    return numpy.asarray(samples).T.tolist()

#  univariate example
# params = [0,0]
# samples = [1,2,3,4,5]
# y = [2,4,6,8,10]

#  multivariate example trivial
# params = [0,0,0]
# samples = [[1,1],[2,2],[3,3],[4,4],[5,5]]
# y = [2,4,6,8,10]


#  multivariate example
params = [0, 0, 0]
samples = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 2], [3, 3], [4, 4]]
y = [2, 4, 6, 8, 10, 2, 5.5, 16]

#Typical values for alfa are 0.1,0.01,0.001,0.0001
alfa = .01  # learning rate

# loop through each sample, adding a 1 to the beginning of each sample if it's a list, otherwise adding a 1 and the sample to a list

for i in range(len(samples)):
    if isinstance(samples[i], list): # if the sample is a list
        samples[i] = [1]+samples[i]
    else: # if the sample is not a list
        samples[i] = [1, samples[i]] # make it a list and add a 1 to the beginning and the sample to the end
print("original samples:")
print(samples)
samples = scaling(samples)
print("scaled samples:")
print(samples)


# Epoch is a hyperparameter that defines the number of times that the learning algorithm will work through the entire training dataset.
# define a limit for epochs
epochs_limit = 1000
epochs = 0

while True:  # run gradient descent until local minima is reached
    oldparams = list(params)
    print(params)
    params = GD(params, samples, y, alfa)
    # only used to show errors, it is not used in calculation
    show_errors(params, samples, y)
    print(params)
    epochs += 1
    # local minima is found when there is no further improvement
    if (oldparams == params or epochs == epochs_limit):
        print("samples: ")
        print(samples)
        print("final params: ")
        print(params)
        break

plt.plot(__errors__)
plt.show()