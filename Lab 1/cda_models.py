import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import SMOTE, ADASYN

from scipy import interp

# Oversampling function to use (SMOTE / ADASYN)
oversampling_func = SMOTE()


def cross_val(X, y, classifier, folds=10):
    
    # Init confmat variables
    TP = FP = TN = FN = 0

    # Split the data and iterate over the folds
    skf = StratifiedKFold(folds)
    for train_index, test_index in skf.split(X, y):
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        
        # .values to get ndarray instead of Series/Index
        y_train = y.iloc[train_index].values
        y_test = y.iloc[test_index].values

        # TODO: PCA (https://arxiv.org/ftp/arxiv/papers/1403/1403.1949.pdf)
        
        # SMOTE
        X_resampled, y_resampled = oversampling_func.fit_resample(X_train, y_train)
        
        # TODO: look into tomek links
        
        # TODO: feature scaling (before or after?)
        
        # Fit model to (resampled) training set
        classifier.fit(X_resampled, y_resampled)
        
        # Predict labels
        y_pred = classifier.predict(X_test)
        
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                if   y_test[i] == 1: TP += 1
                elif y_test[i] == 0: FP += 1
            elif y_pred[i] == 0:
                if   y_test[i] == 0: TN += 1
                elif y_test[i] == 1: FN += 1

    print('   TP: {}'.format(TP))
    print('   FP: {}'.format(FP))
    print('   TN: {}'.format(TN))
    print('   FN: {}'.format(FN))
    print('Total: {}'.format(TP + FP + TN + FN))
    
    
def roc_cross_val(X, y, classifier, folds=10):
    """
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    sfk = StratifiedKFold(folds)
    
    i = 0
    for train_index, test_index in sfk.split(X, y):
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        
        # .values to get ndarray instead of Series/Index
        y_train = y.iloc[train_index].values
        y_test = y.iloc[test_index].values
        
        
        # SMOTE
        X_resampled, y_resampled = oversampling_func.fit_resample(X_train, y_train)
        
        probas_ = classifier.fit(X_resampled, y_resampled).predict_proba(X_test)
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
        
    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Calculate mean ROC / AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    # Fill variance
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    # Plot
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()