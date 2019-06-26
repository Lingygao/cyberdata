import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher

from datetime import datetime

label_encoder = LabelEncoder()
hash_encoder = FeatureHasher(input_type='string')

def process_data(df):
    
    col_hash = ['protocol', 'flags', 'label', 'src_ip', 'dest_ip', ]
    
    for col_name in col_hash:
        df[col_name] = label_encode(df[col_name])
    
    return df

def label_encode(series):
    return label_encoder.fit_transform(series.astype(str))