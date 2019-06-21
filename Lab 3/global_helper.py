import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path, keep_ip=None):
    
    # Rename columns
    column_names = ['date', 'time', 'duration', 'protocol', 'source_ip_port', '->', 'dest_ip_port', 'flags', 'tos', 'packets', 'bytes', 'flows', 'label']
    
    # Read file
    df = pd.read_csv(path, sep="\s+", names=column_names, header=0)
    
    # Parse date / time
    df.set_index(pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S.%f'), inplace=True)
    
    # Split IP:port
    df[['src_ip', 'src_port']] = df['source_ip_port'].str.split(':', 1, expand=True)
    df[['dest_ip', 'dest_port']] = df['dest_ip_port'].str.split(':', 1, expand=True)
    
    # Drop unneeded / leftover columns
    df.drop(columns=['date', 'time', 'source_ip_port', 'dest_ip_port', '->'], inplace=True)
    
    # Only keep rows originating from the specified IP address
    return df[df['src_ip'] == keep_ip] if keep_ip else df