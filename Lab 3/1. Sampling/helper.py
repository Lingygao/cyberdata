import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint


def plot_counts_bar(value_counts, limit=20, save=False):
    
    data = value_counts[:limit]
    
    # Plot values
    ax = data.plot(kind='bar', figsize=(8,4))
    
    # Add bar labels
    for p in ax.patches:
        ax.annotate(str(p.get_height()),  (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0,6), textcoords='offset points')
    
    # Set axis labels
    ax.set_xlabel("IP address")
    ax.set_ylabel("Frequency")
    
    if save: plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0)
        
        
def reservoir_sample(df, m):
    """
    Based on pseudocode at https://en.wikipedia.org/wiki/Reservoir_sampling.
    """
    
    # Keep first m items in memory
    reservoir = df[:m].copy()
    
    # Process stream
    for i in range(m, len(df)):
        
        # Generate random value between 0 and i
        j = randint(0, i)
        
        if j < m: reservoir.iloc[j] = df.iloc[i]
            
    return reservoir

def sample_to_proba(series):
    value_counts = series.value_counts()
    return value_counts / value_counts.sum()