#%%
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import umap
import numpy as np
import pandas as pd
import openpyxl

def autoencoder(features):
    X = features
    input_layer = Input(shape=(X.shape[1],))
    encoded = Dense(5, activation='sigmoid')(input_layer)
    decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)

    autoencoder.fit(X1, Y1,
                    epochs=100,
                    batch_size=300,
                    shuffle=True,
                    verbose = 30,
                    validation_data=(X2, Y2))

    encoder = Model(input_layer, encoded)
    X_hd = encoder.predict(X)
    return X_hd

def umap_algorithm(features):
    fit = umap.UMAP(metric='euclidean',
                n_components=2,
                random_state=0,
                n_neighbors=5,
                min_dist=0.3)
    features_low = fit.fit_transform(features)
    return features_low
# %%
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
    # High dimensions
    X_hd = autoencoder(xi)
    df_X_hd = pd.DataFrame(X_hd, columns = ['x1','x2','x3','x4','x5'])
    df_y_hd = pd.DataFrame(yi, columns=['y1', 'y2'])
    df_X_hd = pd.concat([df_X_hd, df_y_hd], axis = 1)
    df_X_hd.to_excel(r'C:\Users\Asus\Documents\MAJO\Universidad\SEMESTRE 10\INTELIGENCIA ARTIFICIAL\Rice_Dataset_Commeo_and_Osmancik\hd.xlsx', sheet_name='High dimension')

    # Low dimensions
    X_ld = umap_algorithm(xi)
    df_X_ld = pd.DataFrame(X_ld, columns = ['x1','x2'])
    df_y_ld = pd.DataFrame(yi, columns=['y1', 'y2'])
    df_X_ld = pd.concat([df_X_ld, df_y_ld], axis = 1)
    df_X_ld.to_excel(r'C:\Users\Asus\Documents\MAJO\Universidad\SEMESTRE 10\INTELIGENCIA ARTIFICIAL\Rice_Dataset_Commeo_and_Osmancik\ld.xlsx', sheet_name='Low dimension')

# %%
