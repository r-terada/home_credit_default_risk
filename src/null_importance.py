import os
import gc
import sys
import time
import click
import random
import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from pprint import pprint
from functools import reduce
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from config import read_config, KEY_FEATURE_MAP, KEY_MODEL_MAP
from utils import timer
from features.base import Base
from features.stacking import StackingFeaturesWithPasses

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_train_test(conf):
    df = Base.get_df(conf)  # pd.DataFrame

    feature_classes = [KEY_FEATURE_MAP[key] for key in conf.features]
    features = [df]
    for feature in feature_classes:
        with timer(f"load (or create) {feature.__name__}"):
            f = feature.get_df(conf)
            features.append(f)
    with timer("join on SK_ID_CURR"):
        df = reduce(lambda lhs, rhs: lhs.merge(rhs, how='left', on='SK_ID_CURR'), features)
    del features
    gc.collect()

    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    del df
    gc.collect()
    return train_df, test_df


def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data.columns if f not in ([
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
    ])]
    # Go over fold and keep track of CV score (train and valid) and feature importances

    # Shuffle target if required
    y = data['TARGET'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['TARGET'].copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'num_threads': 4,
        'verbose': -1
    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=600)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))

    return imp_df


def score_feature_selection(df=None, train_features=None, target=None):
    # Fit LightGBM
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': .1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 13,
        'num_threads': 4,
        'min_split_gain': .00001,
        'reg_alpha': .00001,
        'reg_lambda': .00001,
        'metric': 'auc'
    }

    # Fit the model
    hist = lgb.cv(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=2000,
        nfold=5,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=500,
        seed=47
    )
    # Return the last mean / std values
    return hist['auc-mean'][-1], hist['auc-stdv'][-1]


@click.command()
@click.option('--config_file', type=str, default='./configs/lgbm_0.json')
def main(config_file):
    np.random.seed(47)
    conf = read_config(config_file)
    print("config:")
    pprint(conf)

    data, _ = get_train_test(conf)

    with timer("calc actual importance"):
        if os.path.exists("misc/actual_imp_df.pkl"):
            actual_imp_df = pd.read_pickle("misc/actual_imp_df.pkl")
        else:
            actual_imp_df = get_feature_importances(data=data, shuffle=False)
            actual_imp_df.to_pickle("misc/actual_imp_df.pkl")

    print(actual_imp_df.head())

    with timer("calc null importance"):
        nb_runs = 100

        if os.path.exists(f"misc/null_imp_df_run{nb_runs}time.pkl"):
            null_imp_df = pd.read_pickle(f"misc/null_imp_df_run{nb_runs}time.pkl")
        else:
            null_imp_df = pd.DataFrame()
            for i in range(nb_runs):
                start = time.time()
                # Get current run importances
                imp_df = get_feature_importances(data=data, shuffle=True)
                imp_df['run'] = i + 1
                # Concat the latest importances with the old ones
                null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
                # Display current run and time used
                spent = (time.time() - start) / 60
                dsp = '\rDone with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
                print(dsp, end='', flush=True)
            null_imp_df.to_pickle("misc/null_imp_df_run{nb_runs}time.pkl")

    print(null_imp_df.head())

    with timer('score features'):
        feature_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
            f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))

        scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
        scores_df.to_pickle("misc/feature_scores_df.pkl")

    with timer('calc correlation'):
        correlation_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
        corr_scores_df.to_pickle("misc/corr_scores_df.pkl")

    with timer('score feature removal by corr_scores'):
        for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99][::-1]:
            with open(f"misc/split_corr_under_threshold_{threshold}.txt", "w") as fp:
                print([_f for _f, _score, _ in correlation_scores if _score < threshold], file=fp)
            with open(f"misc/gain_corr_under_threshold_{threshold}.txt", "w") as fp:
                print([_f for _f, _, _score in correlation_scores if _score < threshold], file=fp)

        for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99][::-1]:
            split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
            gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]

            print('Results for threshold %3d' % threshold)
            print(f'split: use {len(split_feats)} features')
            split_results = score_feature_selection(df=data, train_features=split_feats, target=data['TARGET'])
            print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
            print(f'gain: use {len(gain_feats)} features')
            gain_results = score_feature_selection(df=data, train_features=gain_feats, target=data['TARGET'])
            print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))


if __name__ == '__main__':
    main()
