import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import confusion_matrix


###
# DATA
###

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

def get_test_labels(X_index):
    
    attacks = [
        ('2017-01-16 09:00:00', '2017-01-19 06:00:00'),
        ('2017-01-30 08:00:00', '2017-02-02 00:00:00'),
        ('2017-02-09 03:00:00', '2017-02-10 09:00:00'),
        ('2017-02-12 01:00:00', '2017-02-13 07:00:00'),
        ('2017-02-24 05:00:00', '2017-02-28 08:00:00'),
        ('2017-03-10 14:00:00', '2017-03-13 21:00:00'),
        ('2017-03-25 20:00:00', '2017-03-27 01:00:00')
    ]
    
    y = pd.Series(np.zeros(len(X_index)), index=X_index)
    for date_from, date_to in attacks:
        y.loc[(y.index >= date_from) & (y.index <= date_to)] = 1
        
    return y


###
# PCA
###

def pca_model(X, date_index, **kwargs):
    
    pca = decomposition.PCA(**kwargs)

    # Fit training data
    pca.fit(X)

    # Transform training data
    X_pca = pca.transform(X)

    # Get variance explained
    var_expl = pca.explained_variance_ratio_
    cuml_var = var_expl.cumsum()
    
    # Reconstruct and calculate residuals
    X_reconstruct = pca.inverse_transform(X_pca)
    residuals = pd.Series(__spe(X, X_reconstruct), index=date_index)
    
    return pca, X_reconstruct, residuals, (var_expl, cuml_var)


def pca_pretrained(pca, X, date_index):
    
    # Transform data
    X_pca = pca.transform(X)
    
    # Reconstruct and calculate residuals
    X_reconstruct = pca.inverse_transform(X_pca)
    residuals = pd.Series(__spe(X, X_reconstruct), index=date_index)
    
    return X_reconstruct, residuals
    
    
def __spe(X, X_reconstruct):
    # Calculate (normalized) squared prediction error (error = norm/length of residuals)
    spe = np.linalg.norm(X - X_reconstruct, axis = 1) ** 2
    return spe / max(spe)


def score(labels, predictions):

    tn, fp, fn, tp = score_raw(labels, predictions)
    
    if tp == 0: return (0, 0, 0)
    
    precision =  tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1

def score_raw(labels, predictions):
    return confusion_matrix(labels, predictions).ravel()

###
# PLOTS
###
    
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
    
def plot_residuals(spe, threshold=None, save=False):
    
    fig, ax = plt.subplots(figsize=(10,4))
    
    spe.plot.line(ax=ax)
    
    if threshold:
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (normalized)')
    
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)
    
    plt.show()
    
    
def plot_attacks(df, save=False):
    
    fig, ax = plt.subplots(figsize=(10,2))

    # Plot lines
    df.plot.line(ax=ax, linewidth=1)
    
    plt.fill_between(df.index, df['True'], step="pre", alpha=1)
    plt.fill_between(df.index, df['Predicted'], step="pre", alpha=1)
    
    # Redyce yticks to just 0 and 1
    plt.yticks([0, 1])

    # Add axis labels
    ax.set_xlabel('Date')
    
    # Format x-axis dates
    plt.minorticks_off()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)
    
    # Show figure
    plt.show()