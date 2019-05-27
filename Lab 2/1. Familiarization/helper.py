import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from datetime import timedelta

# PLOT_OFFSET = timedelta(minutes=0)
PLOT_OFFSET = timedelta(minutes=30)

def get_time_slice(df, date_from, date_to):
    datemask = (df.index >= date_from) & (df.index <= date_to)
    return df.loc[datemask].copy()

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

def plot_data(df_full, features=[], figsize=(30,10), save=False):
    
    # Retrieve subset of columns
    df = df_full[features] if len(features) > 0 else df_full
    
    # Scale on/off columns
    df, onoff_cols = scale_onoff(df)
    
    # Plot dataframe (except on/off sensors)
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
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
    plt.legend(handles, __process_labels(labels))
    
    # Add axis labels
    plt.xlabel('Datetime')
    plt.ylabel('Sensor reading')
    
    # Show plot
    if save is not False:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
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
    