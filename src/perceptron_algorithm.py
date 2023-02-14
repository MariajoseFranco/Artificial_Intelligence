# PERCEPTRON ALGORITHM
#%%
import numpy as np
import random

class Perceptron(object):

    def __init__(self, niter, learning_rate, xi, y) -> None:
        self.niter = niter
        self.learning_rate = learning_rate
        self.xi = xi
        self.y = y
        self.m, self.n = xi.shape
        random.seed(10)
        self.wi = self.initializing_w()

    def initializing_w(self):
        wi = np.full((self.n, 1), random.random())
        return wi

    def activation_function(self, xi_row):
        local_field = np.dot(xi_row, self.wi)
        y_hat = self.lineal_function(local_field)
        return y_hat

    def lineal_function(self, local_field):
        y_hat = local_field
        return y_hat

    def sigmoid_function(self, local_field):
        y_hat = 1/(1+np.exp(-local_field))
        return y_hat

    def tanh_function(self, local_field):
        # y_hat = (np.exp(local_field)-np.exp(-local_field))/(np.exp(local_field)+np.exp(-local_field))
        y_hat = np.tanh(local_field)
        return y_hat

    def step_function(self, local_field):
        if local_field > 0:
            y_hat = 1
        else:
            local_field = 0
        return y_hat

    def cost_function(self, y_hat):
        cost = (1/2)*np.sum((self.y - y_hat)**2)
        return cost

    def local_error(self, y, y_hat):
        e = y - y_hat
        return e

    def average_energy_error(self, energy_errors):
        N = len(energy_errors)
        average_energy_error = (1/N)*np.sum(energy_errors) # error cuadratico
        return average_energy_error

    def update_weights(self, xi_row, e):
        delta_w = (e*xi_row).reshape(self.n,1)
        self.wi += self.learning_rate*delta_w
        return self.wi

    def energy_error(self, errors):
        energy_error = (1/2)*np.sum([e**2 for e in errors])
        return energy_error

    def main(self):
        iter_total_errors = []
        for iter in range(self.niter):
            y_hat_list = []
            i = 0
            errors = []
            energy_error_list = []
            # Se hace este for por patron (xi_row --> patron)
            for xi_row in self.xi:
                y_hat = self.activation_function(xi_row)
                cost = self.cost_function(y_hat)
                e = self.local_error(self.y[i], y_hat)
                errors.append(e)
                energy_error = self.energy_error(e)
                energy_error_list.append(energy_error)
                self.wi = self.update_weights(xi_row, e)
                y_hat_list.append(y_hat[0])
                i += 1
            average_energy_error = self.average_energy_error(energy_error_list)
            iter_total_errors.append(average_energy_error)
        return self.wi, y_hat_list, cost, average_energy_error, iter_total_errors

#%%
if __name__ == '__main__':
    niter = 500
    learning_rate = 0.05
    xi = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])
    # OR
    # y = np.array([[0],
    #               [1],
    #               [1],
    #               [1]])

    # XOR
    # y = np.array([[0],
    #               [1],
    #               [1],
    #               [0]])

    # XNOR
    # y = np.array([[1],
    #               [0],
    #               [0],
    #               [1]])

    # AND
    y = np.array([[0],
                  [0],
                  [0],
                  [1]])
    perceptron = Perceptron(niter, learning_rate, xi, y)
    w, y_hat, cost, average_energy_error, iter_total_errors = perceptron.main()
    print('cost: ', cost)
    print('w: ', w)
    print('y_hat: ', y_hat)
    print('average_energy_error: ', average_energy_error)
#%%