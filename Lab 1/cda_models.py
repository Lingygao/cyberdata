import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

def cross_val(X, y, folds, model):
    
    # Init confmat variables
    TP = FP = TN = FN = 0

    # Split the data and iterate over the folds
    kf = StratifiedKFold(folds)
    for train_index, test_index in kf.split(X, y.astype(int)):
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        
        # .values to get ndarray instead of Series/Index
        y_train = y.iloc[train_index].values
        y_test = y.iloc[test_index].values

        # TODO: preprocessing here
        # Feature scaling
        # SMOTE (w/ Tomek links?)
        # PCA?
        
        # Classify training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
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