import pandas as pd
from currency_converter import CurrencyConverter

from sklearn import preprocessing


label_encoder = preprocessing.LabelEncoder()
currency_converter = CurrencyConverter()



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
    
    
    # Label-encode categorical variables
    # N.B.!!! Use categorical --> ordinal ONLY for decision trees, else Minhash
    df['issuercountrycode']            = label_encode(df['issuercountrycode'])
    df['txvariantcode']                = label_encode(df['txvariantcode'])
    df['currencycode']                 = label_encode(df['currencycode'])
    df['shoppercountrycode']           = label_encode(df['shoppercountrycode'])
    df['shopperinteraction']           = label_encode(df['shopperinteraction'])
    df['accountcode']                  = label_encode(df['accountcode'])
    df['cardverificationcodesupplied'] = label_encode(df['cardverificationcodesupplied'])
    df['mail_id']                      = label_encode(df['mail_id'])
    df['ip_id']                        = label_encode(df['ip_id'])
    df['card_id']                      = label_encode(df['card_id'])

    return df, df_raw


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
    

def convert_currency(df):
    return df.apply(lambda x: currency_converter.convert(x['amount'], x['currencycode'], 'USD'), axis=1)


def label_encode(series):
    return label_encoder.fit_transform(series.astype(str))
