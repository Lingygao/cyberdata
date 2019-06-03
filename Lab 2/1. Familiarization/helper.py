import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import seaborn as sns
from datetime import timedelta
from sklearn.metrics import mean_squared_error


# PLOT_OFFSET = timedelta(minutes=0)
PLOT_OFFSET = timedelta(minutes=30)


###
# Familiarization
###

def get_time_slice(df, date_from, date_to):
    datemask = (df.index >= date_from) & (df.index <= date_to)
    return df.loc[datemask].copy()

def corr_heatmap(corr, save=False):
    """Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html"""
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Show plot
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.show()

def scale_onoff(df):
    
    # Get maximum value (+1%)
    scale_val = np.max(df.max(axis=1, numeric_only=True)) * 1.01
    
    # Get on/off columns
    onoff_cols = [col for col in df.columns if col.startswith('S_')]
    
    if len(onoff_cols) > 0:
        # Prevent warnings
        df = df.copy()

        # Scale on/off features w.r.t. max value
        for column in onoff_cols:
            df[column] = df[column] * scale_val
    
    return df, onoff_cols

def plot_data(df_full, features=[], figsize=(30,10), xlabel='Date', save=False):
    
    # Retrieve subset of columns
    df = df_full[features] if len(features) > 0 else df_full
    
    # Scale on/off columns
    df, onoff_cols = scale_onoff(df)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Plot dataframe (except on/off sensors)
    df.loc[:, df.columns.difference(onoff_cols)].plot(ax=ax, use_index=True)
    
    ###
    ## COLOR ON/OFF SENSORS
    
    # Store current legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Store color index (to color multiple on/off sensors differently later)
    color_index = 0
    
    # Iterate on sequences
    for onoff_col in onoff_cols:
        
        current_color = "C"+str(color_index % 9)
        
        # Get column and sort 
        col = df[onoff_col].sort_index()
        
        current = 0
        span_list = []
        
        # Find breakpoints
        for idx,val in col.iteritems():
            if current == 0 and val > 0:
                current = 1
                span_list.append(idx)
            elif current  == 1 and val <= 0:
                current = 0
                span_list.append(idx)
    
        # Plot "sensor on" ranges
        for i in range(0, len(span_list), 2):
            
            # Handle last item possibly out of range
            end = span_list[i+1] if i+1 < len(span_list) else col.index[-1]
            
            # Plot span for sensor "on" (shift with offset for nicer plots)
            plt.axvspan(span_list[i] - PLOT_OFFSET, end - PLOT_OFFSET, alpha=0.2, facecolor=current_color, zorder=0)
            
        # Add legend entry
        patch = mpatches.Patch(color=current_color, alpha=0.4, label=onoff_col)
        handles.append(patch)
        labels.append(onoff_col)
    
        # Increase color index
        color_index = color_index + 1
    
    # Re-apply legend
    plt.legend(handles, __process_labels(labels), loc='center right')
    
    # Add axis labels
    plt.xlabel(xlabel)
    plt.ylabel('Sensor reading')

    # Show plot
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.show()
    
def __process_labels(labels):
    """Rename abbreviations in sensor labels."""
    
    replace_map = {
        'L_': 'Water level ',
        'S_': '[On/Open] ',
        'F_': 'Flow ',
        'P_': 'Pressure ',
        'PU': 'Pump ',
        'J': 'Junction '
    }
    
    # Iterate labels, replace abbreviations
    for i, label in enumerate(labels):
        for abbr, full in replace_map.items():
            label = label.replace(abbr, full)
        labels[i] = label
    
    return labels




###
# Prediction
###

def predict_sliding_windows(df, features=[], window_sizes=[3]):
    
    data = pd.DataFrame()
    results = {}
    
    for feature in features:
        series = df[feature]
        results[feature] = {}
    
        for window in window_sizes:

            # Calculate moving averages
            predictions = series.rolling(window).mean()

            # Calculate errors
            errors = (predictions - series)
            MSE = (errors ** 2).mean()
            MAE = errors.abs().mean()

            # Store results
            data[feature] = series
            data["%s_%d" % (feature, window)] = predictions
            data["%s_%d_err" % (feature, window)] = errors
            
            results[feature][window] = {
                'MSE': MSE,
                'MAE': MAE
            }
    
    return results, data

def plot_prediction(real, predicted, errors, save=False):
    
    fig = plt.figure(figsize=(8,3))
    
    # Create subplots
    ax1 = plt.axes([.1, .3, .8, 1])
    ax1.get_xaxis().set_visible(False)
    ax2 = plt.axes([.1, .1, .8, .2], sharex=ax1)
    
    # Plot real / predicted in upper part
    real.plot.line(linewidth=2,ax=ax1)
    predicted.plot.line(linewidth=2, linestyle='--', ax=ax1)
    
    # Plot errors in lower part
    errors.abs().plot.line(linewidth=2, ax=ax2, color='red')
    
    # Add axis labels
    ax2.set_xlabel('Date')
    ax1.set_ylabel('Sensor reading')
    ax2.set_ylabel('|Error|')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Add grid
    plt.grid()
    
    # Update legend
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    for i in range(len(labels)):
        if labels[i] == real.name: labels[i] = "Actual value"
        elif labels[i] == predicted.name: labels[i] = "Predicted value (window = %s)" % predicted.name[-1:]
    ax1.legend(handles + handles2, labels + ["|Error|"], loc='lower right')
    
    # Show plot
    if save: plt.savefig(save, bbox_inches='tight', pad_inches=0)
    plt.show()