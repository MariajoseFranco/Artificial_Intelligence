import numpy as np

class ActivationFunctions(object):

    def __init__(self) -> None:
        pass

    def lineal_function(self, local_field):
        y_hat = 0.5*local_field + 1/2
        return y_hat

    def lineal_derivate(self, local_field):
        derivate = 0.5
        return derivate

    def sigmoid_function(self, local_field):
        y_hat = 1/(1+np.exp(-local_field))
        return y_hat

    def sigmoid_derivate(self, local_field):
        derivate = self.sigmoid_function(local_field)*(1-self.sigmoid_function(local_field))
        return derivate

    def tanh_function(self, local_field):
        y_hat = np.tanh(local_field)
        return y_hat

    def tanh_derivate(self, local_field):
        derivate = 1-(np.tanh(local_field))**2
        return derivate