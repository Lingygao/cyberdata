import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from scipy import interp

# Ignore data conversion warnings (int -> float is a safe conversion)
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Store categorical columns
categorical_cols = ['issuercountrycode', 'bin', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'cvcresponsecode', 'shopperinteraction', 'accountcode', 'cardverificationcodesupplied', 'mail_id', 'ip_id', 'card_id']
numerical_cols = ['amount']

smote_func = SMOTE()
smote_tomek_func = SMOTETomek()



## DATA PREPARATION FUNCTIONS

def smote(X_train, y_train):
    
    # Oversample (SMOTE) + undersample (remove Tomek links)
    X_train, y_train = smote_func.fit_resample(X_train, y_train)
    
    return X_train, y_train

def smote_tomek(X_train, y_train):
    
    # Oversample (SMOTE) + undersample (remove Tomek links)
    X_train, y_train = smote_tomek_func.fit_resample(X_train, y_train)
    
    return X_train, y_train



# No preprocessing
def prepare_null(X_train, y_train, X_test):
    return X_train, y_train, X_test



# Preprocessing for with / without SMOTE analysis
def prepare_smote_analysis(X_train, y_train, X_test):
    
    # SMOTE
    X_train, y_train = smote(X_train, y_train)
    
    return X_train, y_train, X_test



# Preprocessing for white box algorithm
def prepare_whitebox(X_train, y_train, X_test):
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Scale numerical values
    X_train['amount_dollar'] = preprocessing.scale(X_train['amount_dollar'])
    X_test['amount_dollar'] = preprocessing.scale(X_test['amount_dollar'])
    
    # SMOTE
    X_train, y_train = smote_tomek(X_train, y_train)
    
    return X_train, y_train, X_test



# Preprocessing for black box algorithm
def prepare_blackbox(X_train, y_train, X_test):
    
    # Feature scaling (here or after smote/pca?)
    X_train['amount_dollar'] = preprocessing.scale(X_train['amount_dollar'])
    X_test['amount_dollar'] = preprocessing.scale(X_test['amount_dollar'])

    # PCA (https://arxiv.org/ftp/arxiv/papers/1403/1403.1949.pdf)
    pca = PCA(0.99)
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test  = pca.transform(X_test)
    
    # SMOTE
    X_train, y_train = smote(X_train, y_train)
    
    # TODO: look into tomek links
    
    return X_train, y_train, X_test



def roc_cross_val(X, y, classifier, prep_func, folds=10, caption_msg=""):
    """
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    
    # Init confmat variables
    TP = FP = TN = FN = 0
    precisions = recalls = 0
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig = plt.figure(figsize=(10,6), dpi=72)
    plt.tight_layout()
    
    
    i = 0
    for train_index, test_index in StratifiedKFold(folds, shuffle=True).split(X, y):
        
        ## DATA RETRIEVAL
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        # .values to get ndarray instead of Series/Index
        y_train = y.iloc[train_index].values
        y_test = y.iloc[test_index].values
        
        
        ## CLASSIFICATION
        
        # Manipulate data
        X_train, y_train, X_test = prep_func(X_train, y_train, X_test)
        
        # Fit model to (resampled) training set
        classifier.fit(X_train, y_train)
        
        # Predict labels
        y_pred = classifier.predict(X_test)
        
        # Calculate probabilities
        probas_ = classifier.predict_proba(X_test)
        
        
        ## ROC CURVE
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

        
        ## CONF MAT VALUES
        
        # Compute true/false positives/negatives
        for x in range(len(y_pred)):
            if int(y_pred[x]) == 1:
                if   y_test[x] == 1: TP += 1
                elif y_test[x] == 0: FP += 1
            elif int(y_pred[x]) == 0:
                if   y_test[x] == 0: TN += 1
                elif y_test[x] == 1: FN += 1
            else:
                display(y_pred[x])
                display(int(y_pred[x]))
                display(int(y_pred[x]) == 0)
                display(int(y_pred[x]) == 1)
                
                raise ValueError('WTF?')
        
        i += 1
        
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Calculate mean ROC / AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    # Fill stddev gap
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    # Create plot
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    clfname = type(classifier).__name__
    plt.title('ROC curve - {}{}'.format(clfname, " - {}".format(caption_msg) if caption_msg is not "" else ""))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    
    # Save plot
    params = classifier.get_params()
    if clfname == "KNeighborsClassifier":
        extra_info = params['n_neighbors']
    elif clfname == "RandomForestClassifier":
        extra_info = params['n_estimators']
    else:
        extra_info = ""
    
    plt.savefig("images/{}{}_{}.png".format(type(classifier).__name__, extra_info, prep_func.__name__), dpi=150)
    plt.show()
    
    # Create confusion matrix
    confmat = pd.DataFrame({
        'IsFraud': {'PredFraud': TP, 'PredLegit': FP, 'Total': FP + TP},
        'IsLegit': {'PredFraud': FN, 'PredLegit': TN, 'Total': TN + FN},
        'Total': {'PredFraud': FN + TP, 'PredLegit': TN + FP, 'Total': FP + TP + TN + FN},
    })
    
    # Calculate performance metrics
    TPFN = TP + FN
    TPFP = TP + FP
    recall    = TP / (TP + FN)  if TPFN > 0 else 0
    precision = TP / (TP + FP)  if TPFP > 0 else 0
    
    precisionrecall = precision + recall 
    f1 = 2 * (precision * recall) / (precision + recall) if precisionrecall > 0 else 0
    
    # Return
    return confmat, precision, recall, f1, mean_auc, std_auc

