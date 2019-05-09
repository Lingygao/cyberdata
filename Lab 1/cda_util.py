import pandas as pd
import seaborn as sns

from sklearn import preprocessing

# from currency_converter import CurrencyConverter

label_map = {
    0: 'Legitimate',
    1: 'Fraud'
}

labelenc = preprocessing.LabelEncoder()
# c = CurrencyConverter()

def load_dataset(path):
    
    df = pd.read_csv(path, header=0, index_col=0)

    # Generate labels
    df['labels'] = df['simple_journal'].map({'Chargeback': 1, 'Settled': 0})

    # Drop 'Refused' rows and then the 'simple_journal' col (which is now labels)
    df.drop(df.index[df['simple_journal'] == 'Refused'], inplace=True)
    
#     df.drop(columns=['simple_journal'], inplace=True)
    df.drop(columns=['simple_journal', 'bookingdate', 'creationdate'], inplace=True)

    # Convert datetimes
#     df['bookingdate'] = pd.to_datetime(df['bookingdate'])
#     df['creationdate'] = pd.to_datetime(df['creationdate'])
    
    return df

def cat_to_ord(series):
    cat_series = series.astype("category")
    mapping = dict( enumerate(cat_series.cat.categories ) )
    
    return cat_series.cat.codes, mapping

def label_encode(series):
    return labelenc.fit_transform(series.astype(str))
    

def split_labels(df):
    Y = df['labels']
    X = df.drop(columns='labels')
    return X, Y

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
