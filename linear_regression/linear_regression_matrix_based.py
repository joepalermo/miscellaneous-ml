import numpy as np
import time

# settings ---------------------------------------------------------------------

# print intermediate values of the cost function during learning
print_progress = False
# number of training epochs
num_epochs = 1000
# number of training examples
num_examples = 10000
# lr is the learning rate
lr = 0.1
# m is the number of features
m = 1

# generate training data -------------------------------------------------------
x = np.random.randn(num_examples,m)
y = 2 * np.sum(x, 1) + 1
y = y.reshape((num_examples, 1))

# add column of 1s to x (so h can be represented in terms of dot product)
x = np.hstack((np.ones((x.shape[0], 1), dtype=x.dtype),
                      x))

# define and initialize the model ----------------------------------------------

# initialization of model parameters
theta = np.random.randn(m+1,1)

# the model definition
h = lambda x: np.dot(x, theta)

# train the model --------------------------------------------------------------
print("starting values of theta: " + str(theta))

# define the cost function
c = lambda d: 0.5 * sum((h(x) - y) ** 2 for (x,y) in d)

start_time = time.time()
for i in range(1, num_epochs+1):

    # is there a better way than reshaping?
    gradient = np.sum((h(x) - y) * x, axis=0).reshape((m+1,1))
    theta = theta - lr * gradient/num_examples

    if print_progress:
        print("end of epoch " + str(i) \
            + ", value of cost function: " + str(c(d)))

print("--- %s seconds ---" % (time.time() - start_time))
print("post-training values of theta: " + str(theta))
