import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


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