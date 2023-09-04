
#%%
from multilayer_perceptron_algorithm import MultilayerPerceptron
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


if __name__ == '__main__':
    df = pd.read_excel(r'C:\Users\Asus\Documents\MAJO\Universidad\SEMESTRE 10\INTELIGENCIA ARTIFICIAL\Rice_Dataset_Commeo_and_Osmancik\Rice_Cammeo_Osmancik.xlsx', header=0,  names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "y1", "y2", "y3"])
    df_new = df
    df_new.pop('y3')
    df_normalized=(df_new-df_new.min())/(df_new.max()-df_new.min())
    df_normalized = df_normalized.dropna()
    df_normalized = df_normalized.sample(frac=1)
    x1 = df_normalized["x1"]
    x2 = df_normalized["x2"]
    x3 = df_normalized["x3"]
    x4 = df_normalized["x4"]
    x5 = df_normalized["x5"]
    x6 = df_normalized["x6"]
    x7 = df_normalized["x7"]
    xi = np.transpose(np.array([x1,x2,x3,x4,x5,x6,x7]))
    y1 = df_normalized["y1"]
    y2 = df_normalized["y2"]
    yi = np.transpose(np.array([y1,y2]))
    xi_train, xi_rem, y_train, y_rem = train_test_split(xi,yi, train_size=0.6)
    xi_test, xi_valid, y_test, y_valid = train_test_split(xi_rem,y_rem, train_size=0.5)

    df_average_energy = pd.DataFrame(columns=['Learning Rate', 'Architecture', 'Average Energy'])
    learning_rate = [0.2, 0.5, 0.9]
    num_layers = [3] # 1 capa oculta
    num_hidden_neurons = [i+1 for i in range(6)] # numero de neuronas en las capas ocultas para bajas dimensiones
    # num_hidden_neurons = [i+8 for i in range(6)] # numero de neuronas en las capas ocultas para altas dimensiones
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
                MLperceptron = MultilayerPerceptron(xi_train, xi_train, act_func, niter, lr, nlayers, hidden_neurons, aux_w)
                y_hat, average_energy, w, local_gradient_per_iter, average_error_per_iter = MLperceptron.main()
                train_w.append(w)
                local_gradients_switching_neurons.append(local_gradient_per_iter)
                average_errors_switching_neurons.append(average_error_per_iter)
                df_average_energy = df_average_energy.append({'Learning Rate':lr,
                                                              'Architecture':MLperceptron.layers_size,
                                                              'Average Energy': average_energy}, ignore_index = True)
            MLperceptron.plot_errors(average_errors_switching_neurons, low_dim = True)
# %%
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

nx, mx = xi_train.shape
X = np.empty((nx, mx))
for i in range(len(y_hat)):
    X[i] = y_hat[i].T

# %%
# FOR LOW DIMENSIONS

MLP = MLPRegressor(random_state=1, max_iter=100000)
MLP.fit(X, y_train)
y_predicha = MLP.predict(xi_test)

plt.plot(MLP.loss_curve_)
plt.title('Curva de pérdida para bajas dimensiones')
plt.ylabel('Loss')
plt.xlabel('Epoca')
plt.show()



# %%
# FOR HIGH DIMENSIONS

MLP = MLPRegressor(random_state=1, max_iter=100000)
MLP.fit(X, y_train)
y_predicha = MLP.predict(xi_test)

plt.plot(MLP.loss_curve_)
plt.title('Curva de pérdida para altas dimensiones')
plt.ylabel('Loss')
plt.xlabel('Epoca')
plt.show()
# %%