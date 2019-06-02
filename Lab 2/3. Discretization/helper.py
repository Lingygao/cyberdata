import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import SymbolicAggregateApproximation


def get_time_slice(df, date_from, date_to):
    datemask = (df.index >= date_from) & (df.index <= date_to)
    return df.loc[datemask].copy()


def discretize(df, *args, **kwargs):
    out = pd.DataFrame()
    
    # Discretize each Series in DataFrame
    for sensor in df.columns:
        
        transformed = discretize_series(df[sensor], *args, **kwargs)
        out[sensor] = pd.Series(transformed[0].ravel(), index=df.index)
    
    return out

def discretize_series(series, n_sax_symbols=6, hours_in_segment=4):
    
    # Normalize / rescale series
    data = __normalize_series(series)

    # Determine PAA segment length for # of hours in segment
    n_paa_segments = __get_n_paa_segments(len(series), hours_in_segment)
    
    # SAX (and PAA) transform
    transformed = __sax(data, n_sax_symbols, n_paa_segments)
    
    return transformed

def __sax(data, sax_symbols, paa_segments):
    """
    data: input data
    sax_symbols: number of SAX symbols
    paa_segements: number of PAA segments
    """
    sax = SymbolicAggregateApproximation(alphabet_size_avg=sax_symbols, n_segments=paa_segments)
    return sax.inverse_transform(sax.fit_transform(data))


def plot_discretized_series(og, discr):
    
    # Normalize series
    norm = pd.Series(__normalize_series(og)[0].ravel(), index=og.index)
    
    # Create figure
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    
    # Plot original data
    norm.plot.line(linewidth=2, alpha=0.6)
    
    # Plot discretized data
    discr.plot.line(linewidth=2)
    
    # Add axis labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized sensor readings')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(labels)):
        if labels[i] == 'None': labels[i] = "Original"
        else: labels[i] = "Discretized"
    ax.legend(handles, labels, loc='upper right')
    
    # Show figure
    plt.show()

    
def __normalize_series(series):
    # Rescale series to mean 0 and unit variance
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    return scaler.fit_transform(series)
    
def __get_n_paa_segments(len_data, hours_in_segment):
    return max(2, round(len_data / hours_in_segment))

