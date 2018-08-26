"""
https://www.kaggle.com/aantonova/797-lgbm-and-bayesian-optimization
"""

import gc
import warnings
import pandas as pd
from scipy.stats import ranksums
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


def corr_feature_with_target(feature, target):
    c0 = feature[target == 0].dropna()
    c1 = feature[target == 1].dropna()

    if set(feature.unique()) == set([0, 1]):
        diff = abs(c0.mean(axis=0) - c1.mean(axis=0))
    else:
        diff = abs(c0.median(axis=0) - c1.median(axis=0))

    p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2

    return [diff, p]


def clean_data(data):
    warnings.simplefilter(action='ignore')

    # Removing empty features
    nun = data.nunique()
    empty = list(nun[nun <= 1].index)

    data.drop(empty, axis=1, inplace=True)
    print('After removing empty features there are {0:d} features'.format(data.shape[1]))

    # Removing features with the same distribution on 0 and 1 classes
    corr = pd.DataFrame(index=['diff', 'p'])
    ind = data[data['TARGET'].notnull()].index

    for c in data.columns.drop('TARGET'):
        corr[c] = corr_feature_with_target(data.loc[ind, c], data.loc[ind, 'TARGET'])

    corr = corr.T
    corr['diff_norm'] = abs(corr['diff'] / data.mean(axis=0))

    to_del_1 = corr[((corr['diff'] == 0) & (corr['p'] > .05))].index
    to_del_2 = corr[((corr['diff_norm'] < .5) & (corr['p'] > .05))].drop(to_del_1).index
    to_del = list(to_del_1) + list(to_del_2)
    if 'SK_ID_CURR' in to_del:
        to_del.remove('SK_ID_CURR')

    data.drop(to_del, axis=1, inplace=True)
    print('After removing features with the same distribution on 0 and 1 classes there are {0:d} features'.format(data.shape[1]))

    # Removing features with not the same distribution on train and test datasets
    corr_test = pd.DataFrame(index=['diff', 'p'])
    target = data['TARGET'].notnull().astype(int)

    for c in data.columns.drop('TARGET'):
        corr_test[c] = corr_feature_with_target(data[c], target)

    corr_test = corr_test.T
    corr_test['diff_norm'] = abs(corr_test['diff'] / data.mean(axis=0))

    bad_features = corr_test[((corr_test['p'] < .05) & (corr_test['diff_norm'] > 1))].index
    bad_features = corr.loc[bad_features][corr['diff_norm'] == 0].index

    data.drop(bad_features, axis=1, inplace=True)
    print('After removing features with not the same distribution on train and test datasets there are {0:d} features'.format(data.shape[1]))

    del corr, corr_test
    gc.collect()

    # Removing features not interesting for classifier
    clf = LGBMClassifier(random_state=0)
    train_index = data[data['TARGET'].notnull()].index
    train_columns = data.drop('TARGET', axis=1).columns

    score = 1
    new_columns = []
    while score > .7:
        train_columns = train_columns.drop(new_columns)
        clf.fit(data.loc[train_index, train_columns], data.loc[train_index, 'TARGET'])
        f_imp = pd.Series(clf.feature_importances_, index=train_columns)
        score = roc_auc_score(data.loc[train_index, 'TARGET'],
                              clf.predict_proba(data.loc[train_index, train_columns])[:, 1])
        new_columns = f_imp[f_imp > 0].index

    data.drop(train_columns, axis=1, inplace=True)
    print('After removing features not interesting for classifier there are {0:d} features'.format(data.shape[1]))

    return data
