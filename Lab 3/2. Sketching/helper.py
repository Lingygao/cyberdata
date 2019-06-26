import pandas as pd
import numpy as np
import mmh3
import sys

def count_min_sketch(series, width, depth, normalize=False, show_mem=False):
    
    n = len(series)
    
    # Initialize sketch matrix
    sketch = np.zeros( (depth, width) )
    
    # Generate seeds for hash functions
    seeds = list(range(depth)) # TODO: randomize this?
    
    uniqip = set()
    
    # Process IP addresses
    for _, ip in series.iteritems():
        
        # Add IP to unique IP set
        uniqip.add(ip)
        
        # For each hash function
        for i in range(depth):
            
            # Hash ip and increment cell in matrix
            idx = mmh3.hash(ip, seeds[i]) % width
            sketch[i, idx] += 1
    
    if show_mem:
        return sys.getsizeof(sketch)
    
    # Initialize result series
    result = pd.Series(0, index=uniqip)
    
    # Calculate estimates
    for ip in uniqip:
        
        min_est = n
        
        # For each hash function
        for i in range(depth):
            
            # Hash ip and update min_count if less than already found
            idx = mmh3.hash(ip, seeds[i]) % width
            val = sketch[i, idx]
            if val < min_est: min_est = val
        
        result.at[ip] = min_est
    
    # Normalize w.r.t. data length if needed
    if normalize: result /= n
    
    return result.sort_values(ascending=False)