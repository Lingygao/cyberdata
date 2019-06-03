import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn import preprocessing
from sklearn import decomposition


def load_dataset(path, has_labels=True):
    df = pd.read_csv(path)

    # Convert datetime indices
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H')
    df.set_index('DATETIME', inplace=True)

    labels = None
    if has_labels:

        # Extract labels
        labels = df['ATT_FLAG'].copy()
        df.drop(columns=['ATT_FLAG'], inplace=True)

        # Convert labels to sensable values
        labels[labels != 1] = 0
    
    return df, labels


def get_time_slice(df, date_from, date_to):
    datemask = (df.index >= date_from) & (df.index <= date_to)
    return df.loc[datemask].copy()



def pca_model(X, date_index, y=None, **kwargs):
    
    pca = decomposition.PCA(**kwargs)

    # Fit training data
    pca.fit(X)

    # Transform training data
    X_pca = pca.transform(X)

    # Get variance explained
    var_expl = pca.explained_variance_ratio_
    cuml_var = var_expl.cumsum()
    
    # Calculate residuals
    X_reconstruct = pca.inverse_transform(X_pca)

    # Calculate (normalized) squared prediction error (error = norm/length of residuals)
    spe = np.linalg.norm(X - X_reconstruct, axis = 1) ** 2
    spe /= max(spe)
    residuals = pd.Series(spe, index=date_index)
    
    return X_pca, residuals, (var_expl, cuml_var)


def plot_variance(var_expl, cuml_var):
    
    plt.figure(figsize=(10,6))
    fig, ax = plt.subplots()
    
    ax.bar(range(var_expl.shape[0]), var_expl, alpha = 0.5, align = 'center', color='C3')
    
    ax2 = ax.twinx()
    ax2.step(range(cuml_var.shape[0]), cuml_var, where='mid', color='C0', linewidth=3)
    
    plt.show()
    
def plot_residuals(spe):
    
    plt.figure(figsize=(10,6))
    fig, ax = plt.subplots()
    
    spe.plot.line()
    
    plt.show()