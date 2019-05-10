import pandas as pd
from currency_converter import CurrencyConverter

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher

label_encoder = LabelEncoder()
hash_encoder = FeatureHasher(input_type='string')

currency_converter = CurrencyConverter(fallback_on_wrong_date=True, fallback_on_missing_rate=True)



def get_data(path):

    # Load dataset
    df = load_dataset(path)
    # ['issuercountrycode', 'txvariantcode', 'bin', 'amount', 'currencycode', 'shoppercountrycode', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode', 'mail_id', 'ip_id', 'card_id']
    df_raw = df.copy()
    df.head(2)

    
    ### PREPROCESS
    
    # Convert all amounts to dollars
    # NOTE: this uses the current (?) exchange rate, so it would be better if we could use the exchange rate at the time of the transaction
    df['amount_dollar'] = convert_currency(df)
    
    
    col_hash  = ['issuercountrycode', 'currencycode', 'shoppercountrycode', 'accountcode', 'mail_id', 'ip_id', 'card_id']
    col_label = ['txvariantcode', 'shopperinteraction', 'cardverificationcodesupplied']
    
    for col_name in col_label:
        df[col_name] = label_encode(df[col_name])
    
    for col_name in col_hash:
        df[col_name] = label_encode(df[col_name])
    
    # Dummies: out of memory error
    # FeatureHash: wrong output format

    return df, df_raw


def load_dataset(path):
    
    df = pd.read_csv(path, header=0, index_col=0)

    # Generate labels
    df['labels'] = df['simple_journal'].map({'Chargeback': 1, 'Settled': 0})

    # Drop 'Refused' rows and then the 'simple_journal' col (which is now labels)
    df.drop(df.index[df['simple_journal'] == 'Refused'], inplace=True)
    
    # Drop colums
    df.drop(columns=['simple_journal'], inplace=True)

    df['bin'] = df['bin'].astype(int)
    
    # Convert datetimes
    df['bookingdate'] = pd.to_datetime(df['bookingdate'])
    df['creationdate'] = pd.to_datetime(df['creationdate'])
    
    return df
    

def convert_currency(df):
    return df.apply(lambda x: currency_converter.convert(x['amount'], x['currencycode'], 'USD', date=x['creationdate']), axis=1)


def label_encode(series):
    return label_encoder.fit_transform(series.astype(str))

def hash_encode(series):
    return hash_encoder.fit_transform(series.astype(str))