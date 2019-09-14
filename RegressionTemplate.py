# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np
# more imports

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    # your code
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    # your code
    return theta

# Find thetas using stochiastic gradient descent
# Don't forget to shuffle
def stochiastic_gradient_descent(x, y, learning_rate, num_iterations):
    # your code
    return thetas

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    # your code
    return thetas

# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_iterations, batch_size):
    # your code
    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
   # your code
   return y_predict

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_loss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    x, y = load_data_set('regression-data.txt')
    # plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = gradient_descent(x, y, 0, 0) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = stochiastic_gradient_descent(x, y, 0, 0) # Try different learning rates and number of iterations
    plot(x, y, thetas[-1], "Stochiastic Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Stochiastic Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas = minibatch_gradient_descent(x, y, 0, 0, 0)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Mean Epoch vs Training Loss")
