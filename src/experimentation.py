#%%
from multilayer_perceptron_algorithm import MultilayerPerceptron
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Asus\Documents\MAJO\Universidad\SEMESTRE 10\INTELIGENCIA ARTIFICIAL\DATOS.txt', sep=",", header=None,  names=["x1", "x2", "y"])
    # df = pd.read_excel(file, header=None, names=["x1", "x2", "y"])
    df_new = df[100:400]
    df_normalized=(df_new-df_new.min())/(df_new.max()-df_new.min())
    df_normalized = df_normalized.dropna()
    x1 = df_normalized["x1"]
    x2 = df_normalized["x2"]
    xi = np.transpose(np.array([x1,x2]))
    y = np.array(df_normalized["y"]).reshape(-1,1)
    xi_train, xi_rem, y_train, y_rem = train_test_split(xi,y, train_size=0.6)
    xi_test, xi_valid, y_test, y_valid = train_test_split(xi_rem,y_rem, train_size=0.5)

    df_average_energy = pd.DataFrame(columns=['Learning Rate', 'Architecture', 'Average Energy'])
    learning_rate = [0.2, 0.5, 0.9]
    num_layers = [3, 4, 5] # 1, 2 o 3 capas ocultas
    num_hidden_neurons = [1, 2, 3, 4, 5] # numero de neuronas en las capas ocultas
    niter = 50
    tolerance = 0.002
    act_func = 'sigmoide'
    train_w = []
    aux_w = []
    for lr in learning_rate:
        for nlayers in num_layers:
            local_gradients_switching_neurons = []
            average_errors_switching_neurons = []
            for hidden_neurons in num_hidden_neurons:
                MLperceptron = MultilayerPerceptron(xi_train, y_train, act_func, niter, lr, nlayers, hidden_neurons, aux_w)
                y_hat, average_energy, w, local_gradient_per_iter, average_error_per_iter = MLperceptron.main()
                train_w.append(w)
                local_gradients_switching_neurons.append(local_gradient_per_iter)
                average_errors_switching_neurons.append(average_error_per_iter)
                df_average_energy = df_average_energy.append({'Learning Rate':lr,
                                                              'Architecture':MLperceptron.layers_size,
                                                              'Average Energy': average_energy}, ignore_index = True)
            MLperceptron.plot_local_gradient(local_gradients_switching_neurons)
            MLperceptron.plot_errors(average_errors_switching_neurons)
# %%
