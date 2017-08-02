import numpy as np

# settings ---------------------------------------------------------------------

# use stochastic gradient descent?
sgd = False
# print intermediate values of the cost function during learning
print_progress = False
# number of training epochs
num_epochs = 100
# number of training examples
num_examples = 1000
# lr is the learning rate
lr = 1
# m is the number of features
m = 1

# generate training data -------------------------------------------------------
x_values = np.random.randn(num_examples,m)
y_values = 2 * np.sum(x_values, 1) + 1

# add column of 1s to x_values (so h can be represented in terms of dot product)
x_values = np.hstack((np.ones((x_values.shape[0], 1), dtype=x_values.dtype),
                      x_values))

# wrap features and labels together
d = zip(x_values, y_values)

# define and initialize the model ----------------------------------------------

# initialization of model parameters
theta = np.random.randn(m+1)

# the model definition
h = lambda x: np.dot(theta, x)

# train the model --------------------------------------------------------------
print("starting values of theta: " + str(theta))

# define the cost function
c = lambda d: 0.5 * sum((h(x) - y) ** 2 for (x,y) in d)

for i in range(1, num_epochs+1):
    if sgd:
        for (x,y) in d:
            # update the parameters of the model
            gradient = (h(x) - y) * x
            theta = theta - lr * gradient/num_examples
    else:
        gradient = np.zeros(m+1)
        for (x,y) in d:
            # accumulate gradient values over the training set
            gradient += (h(x) - y) * x
        # update the parameters of the model
        theta = theta - lr * gradient/num_examples
    if print_progress:
        print("end of epoch " + str(i) \
            + ", value of cost function: " + str(c(d)))

print("post-training values of theta: " + str(theta))
