import theano
import theano.tensor as T
import numpy as np
import time

# settings ---------------------------------------------------------------------

# number of training epochs
num_epochs = 1000
# number of training examples
num_examples = 10000
# lr is the learning rate
lr = 0.1
# m is the number of features
m = 1

# generate training data -------------------------------------------------------
x_values = np.random.randn(num_examples,m)
y_values = 2 * np.sum(x_values, 1) + 1

# add column of 1s to x_values (so h can be represented in terms of dot product)
x_values = np.hstack((np.ones((x_values.shape[0], 1), dtype=x_values.dtype),
                      x_values))



# define and initialize the computation ----------------------------------------

# define tensor variables for the input and output of the model
x = T.dmatrix()
y = T.dvector()

# initialization of model parameters
theta = theano.shared(np.random.randn(m+1))

# define the computation graph
h = T.dot(x, theta)
squared_err = T.sqr(h - y)
mse_cost = 0.5 * squared_err.mean()
gradients = T.grad(mse_cost, theta)

# define function to perform training updates
train = theano.function([x,y], updates=[(theta, theta - lr * gradients)])

# train ------------------------------------------------------------------------
print "starting values of theta: " + str(theta.get_value())

start_time = time.time()

for i in xrange(num_epochs):
    train(x_values, y_values)

print("--- %s seconds ---" % (time.time() - start_time))
print "post-training values of theta: " + str(theta.get_value())
