import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from scipy import interp
import datetime

CLASS_LABELS = {"LEGITIMATE":0, "Botnet":1}
CLASS_LABEL_LIST = ["LEGITIMATE", "Botnet"]

def preprocess_df_packets(df, train_test_split=True):
    
    # Remove background traffic
    X = df[df.label != "Background"].copy()
    
    # Rename and extract labels
    y = X["label"].copy().map(CLASS_LABELS)
    X.drop(columns=['label'], inplace=True)
    
    # Reset (date) index and drop IP/date columns
    # TODO? Store IP addresses as int and check if it improves the classification
    X.reset_index(inplace=True, drop=True)
    X.drop(columns=['src_ip', 'dest_ip'], inplace=True)
    
    # Categorical columns to numbers
    # TODO: check if one-hot performs better
    
    X['protocol'] = __str_to_cat(X['protocol'])
    X['flags'] = __str_to_cat(X['flags'])
    
    X['src_port'] = pd.to_numeric(X['src_port'], errors='coerce').fillna(0) # TODO: check 65535 instead of 0
    X['dest_port'] = pd.to_numeric(X['dest_port'], errors='coerce').fillna(0) # TODO: check 65535 instead of 0
    
    # Scale dataset
    X = StandardScaler().fit_transform(X)
    
    # TODO: correct class imbalance
    
    if train_test_split:
        # Return train / test split
        return train_test_split(X, y, test_size=.25, random_state=42)
    else:
        return X, y


def preprocess_df_hosts(df, train_test_split=True):
    
    # Remove background traffic
    X = df[df.label != "Background"].copy()
    
    # Reset (date) index and drop IP/date columns
    # TODO? Store IP addresses as int and check if it improves the classification
    X.reset_index(inplace=True, drop=True)
    
    # Categorical columns to numbers
    X['protocol'] = __str_to_cat(X['protocol'])
    X['flags'] = __str_to_cat(X['flags'])
    
    X['src_port'] = pd.to_numeric(X['src_port'], errors='coerce').fillna(0) # TODO: check 65535 instead of 0
    X['dest_port'] = pd.to_numeric(X['dest_port'], errors='coerce').fillna(0) # TODO: check 65535 instead of 0
    
    # Drop unused columns
    X.drop(columns=['src_port', 'dest_ip', 'dest_port'], inplace=True)
    
    # Rename and extract labels
    X["label"] = X["label"].map(CLASS_LABELS)
    
    X_grp = X.groupby(by='src_ip').agg({
        'duration':  'sum',
#         'protocol':  pd.Series.mode, # mode == most occurring value
#         'flags':     pd.Series.mode,
        'protocol':  lambda x: pd.Series.mode(x)[0], # mode == most occurring value
        'flags':     lambda x: pd.Series.mode(x)[0],
        'tos':       'sum',
        'packets':   'sum',
        'bytes':     'sum',
        'flows':     'sum',
        'label':     'max'
    })
    
    # Rename and extract labels
    y_grp = X_grp["label"].copy()
    X_grp.drop(columns=['label'], inplace=True)
    
    # Scale dataset
    X_grp = StandardScaler().fit_transform(X_grp)
    
    if train_test_split:
        # Return train / test split
        return train_test_split(X_grp, y_grp, test_size=.25, random_state=42)
    else:
        return X_grp, y_grp



def __str_to_cat(series):
    return pd.Categorical(series).codes

def __onehot_encode(df):
    return pd.get_dummies(df)

####
## PLOTS
####

def plot_imbalance(df, classes):
    
    df_classes = df[df["label"].isin(classes)]
    
    sns.countplot('label', data=df_classes)
    
    print("Class imbalance:")
    imb = pd.DataFrame(df_classes['label'].value_counts())
    count_sum = imb['label'].sum()
    imb['percentage'] = imb['label'] / count_sum * 100
    
    with pd.option_context("display.float_format", '{:.2f}'.format):
        display(imb)
        
        
####
## PRETTY PRINT
####

# Source: @jiamingkong - https://gist.github.com/zachguo/10296432
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print(flush=True)
        
        
####
## PLOT
####

def roc_cross_val(X, y, classifier, prep_func=None, folds=10, caption_msg=""):
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
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        ## CLASSIFICATION
        
        if prep_func:
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
    
    plt.savefig("output_figures/{}{}.pdf".format(type(classifier).__name__, extra_info), dpi=150)
    plt.show()
    
    # Create confusion matrix
    confmat = pd.DataFrame({
        'Legitimate': {'Legitimate': TN, 'Botnet': FN, 'Total': TN + FN},
        'Botnet': {'Legitimate': FP, 'Botnet': TP, 'Total': TP + FP},
        'Total': {'Legitimate': TN + FP, 'Botnet': FN + TP, 'Total': FP + TP + TN + FN},
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

