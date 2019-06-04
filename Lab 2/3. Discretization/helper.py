import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from collections import Counter

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import SymbolicAggregateApproximation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

###
# HELPER
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


def discretize(df, *args, **kwargs):
    out = pd.DataFrame()
    
    # Discretize each Series in DataFrame
    for sensor in df.columns:
        transformed, model = discretize_series(df[sensor], *args, **kwargs)
        out[sensor] = pd.Series(transformed[0].ravel(), index=df.index)
    
    return out


def discretize_series(series_raw, n_sax_symbols=6, hours_in_segment=4, inverse_transform=False):
    
    # Copy series
    series = series_raw.copy()
    
    # Normalize / rescale series
    data = normalize_series(series)

    # Determine PAA segment length for # of hours in segment
    n_paa_segments = __get_n_paa_segments(len(series), hours_in_segment)
    
    # SAX (and PAA) transform
    sax = SymbolicAggregateApproximation(alphabet_size_avg=n_sax_symbols, n_segments=n_paa_segments)
    model = sax.fit(data)
    transformed = model.transform(data)
    
    # (Optional) Transform discrete samples back to time series
    if inverse_transform: transformed = model.inverse_transform(transformed)
    
    return transformed, model



def plot_discretized_series(og, discr, save=False):
    
    # Normalize series
    norm = pd.Series(normalize_series(og)[0].ravel(), index=og.index)
    
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
    
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)
    
    # Show figure
    plt.show()

    
def normalize_series(series):
    # Rescale series to mean 0 and unit variance
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    return scaler.fit_transform(series)
    
def __get_n_paa_segments(len_data, hours_in_segment):
    return max(2, round(len_data / hours_in_segment))

def ngram_proba_series(series, n):
    
    # Get n-grams
    ngrams = get_ngrams(series, n)
    
    # Create frequency map
    freq_map = Counter(ngrams)
    
    # Normalize to probabilities
    total_sum = sum(freq_map.values())
    for key in freq_map:
        freq_map[key] /= total_sum
    
    return freq_map
    

def get_ngrams(series, n):
    
    # Generate n-grams
    # Source: http://www.albertauyeung.com/post/generating-ngrams-python/
    return zip(*[series[i:] for i in range(n)])

    

def detect(X, paa, sax, ngram, threshold):
    
    # Set prediction vector to maximum length (multiple of paa segment size)
    predictions = [0] * (len(X) - len(X) % paa)
    
    # Iterate sensors
    for sensor in X.columns:
    
        # Discretize training data
        X_discr_train, _ = discretize_series(X[sensor], n_sax_symbols=sax, hours_in_segment=paa)
        X_discr_train = X_discr_train[0].ravel()

        # Get n-gram probabilities
        proba = ngram_proba_series(X_discr_train, n=ngram)
        
        # Find anomalies
        labels = find_anomalies(X_discr_train, proba, n=ngram, threshold=threshold, window_size=paa)
        
        maxlen = min(len(predictions), len(labels))
        
        # Add alarms to overall predictions
        predictions = np.add(predictions[:maxlen], labels[:maxlen])
    
    # Reset incidents which have been predicted by multiple sensors
    predictions[predictions > 1] = 1
    
    return predictions


def find_anomalies(X, proba, n, threshold, window_size):
    
    limit = len(X)*window_size
    y = [0] * limit
    
    count = 0
    for ngram in get_ngrams(X, n):
        
        # Check n-gram probability
        p = proba.get(ngram, 0)
        
        if p == 0:
            print('Hm')
        
        # Raise alarm if p < threshold
        if p < threshold:
            
            # Calculate original dataset indices (start and end, transform back with PAA segment size)
            idx_start = count * window_size
            idx_end   = idx_start + window_size
            
            for i in range(idx_start, idx_end):
                if i >= limit: break;
                y[i] = 1
            
        count += 1
    
    return y


def score(labels, predictions):

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    if tp == 0: return (0, 0, 0)
    
    precision =  tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1
        