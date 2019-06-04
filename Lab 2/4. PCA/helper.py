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
    
    return pca, X_reconstruct, residuals, (var_expl, cuml_var)


def plot_variance(var_expl, cuml_var, save=False):
    
    fig, ax = plt.subplots(figsize=(8,4))
    
    ax.bar(range(1,var_expl.shape[0]+1), var_expl, alpha = 0.5, align = 'center', color='C1')
    
    ax2 = ax.twinx()
    ax2.step(range(1, cuml_var.shape[0]+1), cuml_var, where='mid', color='C0', linewidth=3)
    ax2.set_ylim([0,1.05])
    
    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Variance explained')
    ax2.set_ylabel('Cumulative variance explained')
    
    plt.xticks(np.arange(1, var_expl.shape[0]+1, 1.0))
    
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)
    
    plt.show()
    
def plot_residuals(spe, save=False):
    
    fig, ax = plt.subplots(figsize=(10,4))
    
    spe.plot.line(ax=ax)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (normalized)')
    
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)
    
    plt.show()