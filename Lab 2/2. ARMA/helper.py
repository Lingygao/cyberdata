import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import confusion_matrix


def get_time_slice(df, date_from, date_to):
    datemask = (df.index >= date_from) & (df.index <= date_to)
    return df.loc[datemask].copy()

#The below function was borrowed from Yair Beer on StackOverflow. It is basically a modified version of autocorrelation_plot,
# which allows to also pass the amount of the amount of samples, which is makes plotting a lot easier. My personal addition
# was to set the default of n_samples to 10 instead of None. For more information, please see:
# https://stackoverflow.com/questions/38503381/set-number-of-lags-in-python-pandas-autocorrelation-plot, visited 03-06-2019. 
from pandas.compat import lmap


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


def autocorrelation_plot(series, n_samples=10,ax=None, **kwds):
    """Autocorrelation plot for time series.

    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method

    Returns:
    -----------
    ax: Matplotlib axis object
    """
    import matplotlib.pyplot as plt
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(1, n_samples), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = (np.arange(n) + 1).astype(int)
    y = lmap(r, x)
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey')
    ax.axhline(y=z95 / np.sqrt(n), color='grey')
    ax.axhline(y=0.0, color='black')
    ax.axhline(y=-z95 / np.sqrt(n), color='grey')
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    if n_samples:
        ax.plot(x[:n_samples], y[:n_samples], **kwds)
    else:
        ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax
