import os
import gc
import sys
import click
import pickle
import random
import sklearn
import hyperopt
import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK

from config import read_config, KEY_FEATURE_MAP, KEY_MODEL_MAP
from utils import timer, reduce_mem_usage
from models import LightGBM
from features.base import Base
from features.feature_cleaner import clean_data
from features.stacking import StackingFeaturesWithPasses

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_train_test(conf):
    df = Base.get_df(conf)  # pd.DataFrame

    feature_classes = [KEY_FEATURE_MAP[key] for key in conf.features]
    for feature in feature_classes:
        with timer(f"process {feature.__name__}"):
            f = feature.get_df(conf)
            if conf.options.drop_duplicate_column_on_merge:
                cols_to_drop = [c for c in f.columns if (c in df.columns) and (c != 'SK_ID_CURR')]
                if cols_to_drop:
                    print(f"drop columns: {cols_to_drop}")
                    f = f.drop(cols_to_drop, axis=1)
            if "reduce_mem_usage" in conf.options and conf.options.reduce_mem_usage:
                with timer("reduce_mem_usaga"):
                    f = reduce_mem_usage(f)
            df = df.merge(f, how='left', on='SK_ID_CURR')
            del f
            gc.collect()

    if "stacking_features" in conf:
        StackingFeaturesWithPasses.set_result_dirs(conf.stacking_features)
        f = StackingFeaturesWithPasses.get_df(conf)
        df = df.merge(f, how='left', on='SK_ID_CURR')

    if "drop_features_list_file" in conf.options:
        with open(conf.options.drop_features_list_file, "r") as fp:
            line = fp.read()
            feature_to_drop = eval(line)
        print(f"drop columns in {conf.options.drop_features_list_file}")
        df = df.drop(feature_to_drop, axis=1)

    if "clean_data" in conf.options and conf.options.clean_data:
        with timer("clean_data"):
            df = clean_data(df)

    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    del df
    gc.collect()
    return train_df, test_df


@click.command()
@click.option('--config_file', type=str, default='./configs/lgbm_0.json')
@click.option('--num_opt_eval', type=int, default=200)
def main(config_file, num_opt_eval):
    conf = read_config(config_file)

    train_df, test_df = get_train_test(conf)
    feats = [f for f in train_df.columns if f not in ([
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
    ])]
    print(f"use {len(feats)} features.")

    parameter_space = {
        "learning_rate": hp.quniform("learning_rate", 0.01, 0.02, 0.001),
        "max_bin": hp.quniform("max_bin", 5, 500, 5),
        "num_leaves": hp.quniform("num_leaves", 16, 128, 1),
        "min_child_samples": hp.quniform("min_child_samples", 1, 300, 1),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.01, 0.2, 0.01),
        "subsample": hp.quniform("subsample", 0.01, 0.8, 0.01),
        "min_gain_to_split": hp.quniform("min_gain_to_split", 0.0, 5.0, 0.1),
        "reg_alpha": hp.quniform("reg_alpha", 0, 100, 0.1),
        "reg_lambda": hp.quniform("reg_lambda", 0, 100, 0.1)
    }

    def objective(params):
        conf.model = {
            **conf.model,
            "clf_params": {
                "learning_rate": float(params["learning_rate"]),
                "max_bin": int(params["max_bin"]),
                "max_depth": int(params["max_depth"]),
                "num_leaves": int(params["num_leaves"]),
                "min_child_samples": int(params["min_child_samples"]),
                "colsample_bytree": float(params["colsample_bytree"]),
                "subsample": float(params["subsample"]),
                "min_gain_to_split": float(params["min_gain_to_split"]),
                "reg_alpha": float(params["reg_alpha"]),
                "reg_lambda": float(params["reg_lambda"]),
                "n_estimators": 10000,
                "max_depth": -1,
                "nthread": -1,
                "scale_pos_weight": 1,
                "is_unbalance": False,
                "silent": -1,
                "verbose": -1,
                "random_state": 0
            }
        }
        pprint(conf.model.clf_params)

        model = LightGBM()
        score = model.train_and_predict_kfold(
            train_df,
            test_df,
            feats,
            'TARGET',
            conf
        )
        return {'loss': -1.0 * score, 'status': STATUS_OK}

    print("====== optimize lgbm parameters ======")
    trials = hyperopt.Trials()
    best = fmin(objective, parameter_space, algo=tpe.suggest,
                max_evals=num_opt_eval, trials=trials, verbose=1)
    print("====== best estimate parameters ======")
    pprint(best)
    # for key, val in best.items():
    #     print(f"    {key}: {val}")
    print("============= best score =============")
    best_score = -1.0 * trials.best_trial['result']['loss']
    print(best_score)
    pickle.dump(trials.trials, open(
        os.path.join(
            conf.dataset.output_directory,
            f'{datetime.now().strftime("%m%d%H%M")}_{conf.config_file_name}_trials_score{best_score}.pkl'
        ), 'wb')
    )


if __name__ == '__main__':
    main()
