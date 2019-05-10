import pandas as pd

label_map = {
    0: 'Legitimate',
    1: 'Fraud'
}


def cat_to_ord(series):
    cat_series = series.astype("category")
    mapping = dict( enumerate(cat_series.cat.categories ) )
    
    return cat_series.cat.codes, mapping

def label_encode(series):
    return label_encoder.fit_transform(series.astype(str))
    

def split_labels(df):
    y = df['labels']
    X = df.drop(columns='labels')
    
    return X, y

def get_class_balance(df):
    
    # Group
    grp = df.groupby('labels').size().reset_index()
    
    # Calculate counts / percentages
    grp.columns = ['label', 'count']
    grp['pct'] = grp['count']*100/(sum(grp['count']))
    grp.set_index('label', inplace=True)
    
    # Map index from 0/1 to names
    grp.index = grp.index.map(label_map)
    grp.index.name = ""
    
    # Add totals
    grp.loc["Total"] = grp.sum()
    
    # Formatting
    grp['count'] = grp['count'].astype(int)
    
    return grp
