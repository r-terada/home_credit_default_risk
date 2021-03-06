import os
import gc
import sys
import click
import random
import sklearn
import numpy as np
import pandas as pd
from pprint import pprint

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
            if "drop_duplicate_column_on_merge" in conf.options and conf.options.drop_duplicate_column_on_merge:
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
@click.option('--num_experiment', type=int, default=100)
def main(config_file, num_experiment):
    conf = read_config(config_file)
    print("config:")
    pprint(conf)

    train_df, test_df = get_train_test(conf)
    feats = [f for f in train_df.columns if f not in ([
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
    ])]
    print(f"{len(feats)} features.")

    for i in range(num_experiment):
        feats_to_use = [f for f in feats if random.random() < conf.feature_sample_rate]
        print(f"use {len(feats_to_use)} features.")
        model = KEY_MODEL_MAP[conf.model.name]()
        with timer(f"train with {model.__class__.__name__}. use {conf.feature_sample_rate * 100} % of features"):
            model.train_and_predict_kfold(
                train_df,
                test_df,
                feats_to_use,
                'TARGET',
                conf
            )


if __name__ == '__main__':
    main()
