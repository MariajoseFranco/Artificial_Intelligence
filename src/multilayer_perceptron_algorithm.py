# MULTILAYER PERCEPTRON ALGORITHM
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Union
from activation_functions import ActivationFunctions
from sklearn.model_selection import train_test_split

class MultilayerPerceptron(object):

    def __init__(self, xi, y, act_func, niter: int, learning_rate: List[float], nlayers: int, hidden_neurons, w) -> None:
        self.niter = niter
        self.learning_rate = learning_rate
        self.nlayers = nlayers
        self.hidden_neurons = hidden_neurons
        self.xi = xi
        self.y = y
        self.m, self.n = self.xi.shape
        _, self.l = self.y.shape
        self.layers_size = self.define_layers_size()
        if w == []:
            self.wi = self.initializing_w()
        else:
            self.wi = self.initializing_w(True)
        self.bias = self.initializing_bias()
        self.act_func = [act_func]*self.nlayers
        self.AF = ActivationFunctions()
        # np.random.seed(10)

    def initializing_w(self, train_w = False):
        wi_list = []
        input = self.xi[0].reshape(1,self.n)
        _, n_inputs = input.shape
        if train_w == False:
            for layer in range(self.nlayers):
                wi = np.random.rand(n_inputs, self.layers_size[layer])
                n_inputs = self.layers_size[layer]
                wi_list.append(wi)
        else:
            for layer in range(self.nlayers):
                wi = np.random.rand(n_inputs, self.layers_size[layer])
                n_inputs = self.layers_size[layer]
                wi_list.append(wi)
        return wi_list

    def initializing_bias(self):
        bias_list = []
        for layer in range(self.nlayers):
            bias = np.random.rand(self.layers_size[layer], 1)*0
            bias_list.append(bias)
        return bias_list

    def define_layers_size(self):
        num_hidden_neurons = self.nlayers-2
        hidden_layers_neurons = [self.hidden_neurons]*num_hidden_neurons
        hidden_layers_neurons.insert(0, self.n)
        hidden_layers_neurons.extend([self.l])
        return hidden_layers_neurons

    def forward_propagation(self, xi_row: np.array) -> Union[np.array, np.array, np.array]:
        inputs = xi_row.reshape(1,self.n)
        # Se hace este for por capa
        local_field_list = []
        outputs_list = []
        for layer in range(self.nlayers):
            wi = self.wi[layer]
            bias = self.bias[layer]
            act_func = self.act_func[layer]
            y_hat, local_field = self.activation_function(act_func, inputs, wi, bias)
            outputs_list.append(y_hat)
            local_field_list.append(local_field)
            inputs = y_hat
        return outputs_list, local_field_list

    def backward_propagation(self, local_field_layers, local_gradient, output_per_layer, xi_row):
        inputs = xi_row.reshape(1,self.n)
        delta_w_list = []
        delta_bias_list = []
        local_gradients_layers = [local_gradient]
        # Para la capa de salida
        Yj = output_per_layer[-2]
        delta_w = self.delta_w(local_gradient, Yj)
        delta_bias = local_gradient
        delta_w_list.insert(0, delta_w.T)
        delta_bias_list.insert(0, delta_bias.T)
        # Para el resto de las capas
        for layer in reversed(range(self.nlayers-1)):
            local_field = local_field_layers[layer]
            delta_bias = self.delta_bias_hidden_layer(layer, local_field, local_gradient)
            local_gradient = self.local_gradient_hidden_layer(layer, local_field, local_gradient)
            delta_bias_list.insert(0, delta_bias.T)
            local_gradients_layers.insert(0, local_gradient)
            Yj = output_per_layer[layer-1]
            if layer == 0:
                Yj = inputs
            delta_w = self.delta_w(local_gradient, Yj)
            delta_w_list.insert(0, delta_w.T)
        return delta_w_list, local_gradients_layers, delta_bias_list

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

    def activation_function(self, activation_function, xi_row, wi, bias):
        local_field = np.dot(xi_row, wi) + bias.T
        if activation_function == 'lineal':
            y_hat = self.AF.lineal_function(local_field)
        elif activation_function == 'sigmoide':
            y_hat = self.AF.sigmoid_function(local_field)
        elif activation_function == 'tanh':
            y_hat = self.AF.tanh_function(local_field)
        return y_hat, local_field

    def update_weights(self, delta_w, learning_rate):
        for layer in range(self.nlayers):
            self.wi[layer] += learning_rate*delta_w[layer]
        return self.wi

    def update_bias(self, delta_bias, learning_rate):
        for layer in range(self.nlayers):
            self.bias[layer] += learning_rate*delta_bias[layer]
        return self.bias

    def delta_w(self, local_gradient, Yj):
        delta_w = np.dot(local_gradient.T, Yj)
        return delta_w

    def average_delta_w(self, delta_w_list):
        N = len(delta_w_list)
        average_dw_list = []
        for layer in range(self.nlayers):
            dw = np.zeros(delta_w_list[0][layer].shape)
            for patron in range(N):
                dw = dw + delta_w_list[patron][layer]
            avg_dw = dw
            average_dw_list.append(avg_dw)
        return average_dw_list

    def local_gradient_output_layer(self, error, local_field_k):
        if self.act_func[-1] == 'lineal':
            local_gradient_k = error*self.AF.lineal_derivate(local_field_k)
        elif self.act_func[-1] == 'sigmoide':
            local_gradient_k = error*self.AF.sigmoid_derivate(local_field_k)
        elif self.act_func[-1] == 'tanh':
            local_gradient_k = error*self.AF.tanh_derivate(local_field_k)
        return local_gradient_k

    def local_gradient_hidden_layer(self, layer, local_field, local_gradient_k):
        w_kj = self.wi[layer+1]
        m,n = w_kj.shape
        sumatoria = np.zeros((1, m))
        for i in range(n):
            wkj = w_kj[:, i]
            local_gradient = local_gradient_k[0][i]
            product = local_gradient*wkj
            sumatoria += product
        if self.act_func[layer] == 'lineal':
            local_field_derivate = self.AF.lineal_derivate(local_field)
        elif self.act_func[layer] == 'sigmoide':
            local_field_derivate = self.AF.sigmoid_derivate(local_field)
        elif self.act_func[layer] == 'tanh':
            local_field_derivate = self.AF.tanh_derivate(local_field)
        local_gradient_j = local_field_derivate*sumatoria
        return local_gradient_j

    def delta_bias_hidden_layer(self, layer, local_field, local_gradient_k):
        bias_kj = self.bias[layer+1]
        m,n = bias_kj.shape
        sumatoria = np.zeros((1, n))
        for i in range(m):
            biaskj = bias_kj.T[:, i]
            local_gradient = local_gradient_k[0][i]
            product = local_gradient*biaskj
            sumatoria += product
        if self.act_func[layer] == 'lineal':
            local_field_derivate = self.AF.lineal_derivate(local_field)
        elif self.act_func[layer] == 'sigmoide':
            local_field_derivate = self.AF.sigmoid_derivate(local_field)
        elif self.act_func[layer] == 'tanh':
            local_field_derivate = self.AF.tanh_derivate(local_field)
        delta_bias = local_field_derivate*sumatoria
        return delta_bias

    def average_local_gradients(self, local_gradient_list):
        suma = 0
        suma_aux = []
        m = len(local_gradient_list[0])
        n = len(local_gradient_list)
        for i in range(m):
            for patron in local_gradient_list:
                suma += patron[0][i]
            suma_aux.append(suma)
        average_local_gradients = (1/n)*np.array(suma_aux)
        return average_local_gradients

    def plot_local_gradient(self, local_gradient_per_iter):
        plt.title('Gradiente local con función de activación ' + self.act_func[-1] + ', lr = ' + str(self.learning_rate) + ' y ' + str(self.nlayers-2) + ' capas ocultas')
        plt.ylabel('Gradiente local')
        plt.xlabel('Epoca')
        plt.plot(local_gradient_per_iter[0], label = '1 neuron in hidden layer')
        plt.plot(local_gradient_per_iter[1], label = '2 neuron in hidden layer')
        plt.plot(local_gradient_per_iter[2], label = '3 neuron in hidden layer')
        plt.plot(local_gradient_per_iter[3], label = '4 neuron in hidden layer')
        plt.plot(local_gradient_per_iter[4], label = '5 neuron in hidden layer')
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_errors(self, errors):
        plt.title('Errores promedio con función de activación ' + self.act_func[-1] + ', lr = ' + str(self.learning_rate) + ' y ' + str(self.nlayers-2) + ' capas ocultas')
        plt.ylabel('Errors')
        plt.xlabel('Epoca')
        plt.plot(errors[0], label = '1 neuron in hidden layer')
        plt.plot(errors[1], label = '2 neuron in hidden layer')
        plt.plot(errors[2], label = '3 neuron in hidden layer')
        plt.plot(errors[3], label = '4 neuron in hidden layer')
        plt.plot(errors[4], label = '5 neuron in hidden layer')
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_y_hat_vs_y_real(self, y_hat):
        m = len(y_hat)
        n = len(y_hat[0][0])
        y_hat = np.array(y_hat).reshape(m,n)
        y_real = self.y
        plt.title('y_real vs y_estimada')
        plt.ylabel('y')
        plt.xlabel('Epoca')
        plt.plot(y_hat, label= "y_estimada")
        plt.plot(y_real, label= "y_real")
        plt.tight_layout()
        plt.legend()
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
            delta_bias_list = []
            local_gradients_layers_list = []
            # Se hace este for por patron (xi_row --> patron)
            for xi_row in self.xi:
                output_per_layer, local_field_per_layer = self.forward_propagation(xi_row)
                # Para la capa de salida
                y_hat_k = output_per_layer[-1]
                y_hat_list.append(y_hat_k)
                local_field_k = local_field_per_layer[-1]

                error = self.output_error(self.y[patron], y_hat_k)
                error_list.append(error)

                local_gradient_k = self.local_gradient_output_layer(error, local_field_k)
                local_gradient_list.append(local_gradient_k)

                inst_energy = self.error_instant_energy(error_list[patron])
                instant_energy.append(inst_energy)

                delta_w_per_layer, local_gradients_layers, delta_bias_per_layer = self.backward_propagation(local_field_per_layer, local_gradient_k, output_per_layer, xi_row)
                delta_w_list.append(delta_w_per_layer)
                local_gradients_layers_list.append(local_gradients_layers)
                delta_bias_list.append(delta_bias_per_layer)

                patron += 1
            # Promedios
            average_energy = np.mean(instant_energy)
            average_local_gradient = self.average_local_gradients(local_gradient_list)
            average_delta_w = self.average_delta_w(delta_w_list)
            average_delta_bias = self.average_delta_w(delta_bias_list)

            average_error_per_iter.append(average_energy)
            local_gradient_per_iter.append(average_local_gradient)

            self.wi = self.update_weights(average_delta_w, self.learning_rate)
            self.bias = self.update_bias(average_delta_bias, self.learning_rate)
        return y_hat_list, average_energy, self.wi, local_gradient_per_iter, average_error_per_iter