# MULTILAYER PERCEPTRON ALGORITHM
#%%
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from typing import List, Union
from activation_functions import ActivationFunctions as AF

class MultilayerPerceptron(object):

    def __init__(self, act_func, niter: int, nlayers: int, layers_size: List[int], xi: np.array, y: np.array) -> None:
        self.niter = niter
        self.nlayers = nlayers
        self.layers_size = layers_size
        self.xi = xi
        self.y = y
        self.m, self.n = xi.shape
        self.wi = self.initializing_w()
        self.act_func = act_func
        # random.seed(10)

    def initializing_w(self) -> List[float]:
        wi_list = []
        input = self.xi[0].reshape(1,self.n)
        _, n_inputs = input.shape
        for layer in range(self.nlayers):
            wi = np.full((n_inputs, self.layers_size[layer]), random.random())
            n_inputs = self.layers_size[layer]
            wi_list.append(wi)
        return wi_list

    def forward_propagation(self, xi_row: np.array) -> Union[np.array, np.array, np.array]:
        inputs = xi_row.reshape(1,self.n)
        # Se hace este for por capa
        for layer in range(self.nlayers):
            wi = self.wi[layer]
            y_hat, local_field = self.activation_function(inputs, wi)
            last_inputs = inputs
            inputs = y_hat
        return y_hat, local_field, last_inputs

    def back_propagation(self):
        pass

    def output_error(self, y, y_hat):
        y = y.reshape(1,y.shape[0])
        e = y - y_hat
        return e

    def error_instant_energy(self, errors):
        energy = (1/2)*np.sum([e**2 for e in errors])
        return energy

    def average_energy(self, instant_energy):
        N = len(instant_energy)
        average_energy = (1/N)*np.sum(instant_energy)
        return average_energy

    def activation_function(self, xi_row, wi):
        local_field = np.dot(xi_row, wi)
        if self.act_func == 'lineal':
            y_hat = self.lineal_function(local_field)
        elif self.act_func == 'sigmoide':
            y_hat = self.sigmoid_function(local_field)
        elif self.act_func == 'tanh':
            y_hat = self.tanh_function(local_field)
        return y_hat, local_field

    # # FUNCIONES DE ACTIVACION
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
    # # FUNCIONES DE ACTIVACION

    def update_weights(self, delta_w):
        w = self.wi[-1]
        w += delta_w
        self.wi[-1] = w
        return w

    def delta_w(self, errors, local_gradient):
        delta_w = np.dot(errors, local_gradient.T)
        return delta_w

    def local_gradient(self, local_field, Yj):
        if self.act_func == 'lineal':
            local_gradient = np.dot(self.lineal_derivate(local_field), Yj.T)
        elif self.act_func == 'sigmoide':
            local_gradient = np.dot(self.sigmoid_derivate(local_field), Yj.T)
        elif self.act_func == 'tanh':
            local_gradient = np.dot(self.tanh_derivate(local_field), Yj.T)
        return local_gradient

    def plot_local_gradient(self, local_gradient_per_iter):
        epoca = [iter for iter in range(self.niter)]
        plt.title('Gradiente local con función de activación lineal')
        plt.ylabel('Gradiente local')
        plt.xlabel('Epoca')
        plt.plot(epoca, local_gradient_per_iter)
        plt.show()

    def plot_errors(self, errors):
        epoca = [iter for iter in range(self.niter)]
        plt.title('Errores promedio para cada epoca')
        plt.ylabel('Errors')
        plt.xlabel('Epoca')
        plt.plot(epoca, errors)
        plt.show()

    def main(self):
        average_error_per_iter = []
        local_gradient_per_iter = []
        for iter in range(self.niter): # iter = epoca
            patron = 0
            y_hat_list = []
            instant_energy = []
            error_list = []
            local_gradient_list = []
            delta_w_list = []
            # Se hace este for por patron (xi_row --> patron)
            for xi_row in self.xi:
                y_hat, local_field, Yj = self.forward_propagation(xi_row)
                y_hat_list.append(y_hat)

                error = self.output_error(self.y[patron], y_hat)
                error_list.append(error)

                local_gradient = self.local_gradient(local_field, Yj)
                local_gradient_list.append(local_gradient)

                inst_energy = self.error_instant_energy(error_list[patron])
                instant_energy.append(inst_energy)

                delta_w = self.delta_w(error, local_gradient)
                delta_w_list.append(delta_w)

                patron += 1
            # Promedios
            average_energy = np.mean(instant_energy)
            average_error = np.mean(error_list)
            average_local_gradient = np.mean(local_gradient_list)
            average_delta_w = np.mean(delta_w_list)

            average_error_per_iter.append(average_error)
            local_gradient_per_iter.append(average_local_gradient)

            w = self.update_weights(average_delta_w)
        # Plots
        self.plot_local_gradient(local_gradient_per_iter)
        self.plot_errors(average_error_per_iter)
        return y_hat_list, average_energy, average_error, w

#%%
if __name__ == '__main__':
    # Cargar datos
    df = pd.read_csv(r'C:\Users\Asus\Documents\MAJO\Universidad\SEMESTRE 10\INTELIGENCIA ARTIFICIAL\DATOS.txt', sep=",", header=None,  names=["x1", "x2", "y"])
    datos = df[0:300]
    x1 = datos["x1"]
    x2 = datos["x2"]
    xi = np.transpose(np.array([x1,x2]))
    y = np.array(datos["y"]).reshape(-1,1)

    niter = 200
    nlayers = 3
    layers_size = [2,1,1]
    act_func = 'tanh'

    MLperceptron = MultilayerPerceptron(act_func, niter, nlayers, layers_size, xi, y)
    y_hat, average_energy, average_error, w = MLperceptron.main()
    print('y_hat: ', y_hat)
    print('w: ', w)
    print('average_energy: ', average_energy)
    print('average_error: ', average_error)
#%%
    # xi = np.array([[0,0],
    #                [0,1],
    #                [1,0],
    #                [1,1]])
    # xi = np.array([[1,2],
    #                [3,4],
    #                [5,6],
    #                [7,8]])
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
    # y = np.array([[0],
    #               [0],
    #               [0],
    #               [1]])

    # y = np.array([[1],
    #               [0],
    #               [0],
    #               [0]])
    # y = np.array([[2],
    #               [5],
    #               [7],
    #               [10]])
#%%